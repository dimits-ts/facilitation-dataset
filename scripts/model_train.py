from pathlib import Path
import argparse
import random

import pandas as pd
import torch
import transformers

import util.classification


GRAD_ACC_STEPS = 1
EVAL_STEPS = 400
EPOCHS = 30
EARLY_STOP_WARMUP = 1000
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 5
FINETUNE_ONLY_HEAD = True
MODEL = "answerdotai/ModernBERT-base"


# ───────────────────────────────── Dataset for *training* ───────────────────
class DiscussionModerationDataset(torch.utils.data.Dataset):
    """
    Map-style dataset that, on-the-fly, builds the
    "<CONTEXT> … <COMMENT> …" sequence *and* returns the label.

    • Text is tokenised lazily so no big tensor object sits in RAM.
    • Outputs are Python lists; a DataCollatorWithPadding will
      pad and convert to tensors batch-wise.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # fast look-up:  message_id → raw text
        self._id2text = self.df.set_index("message_id")["text"].to_dict()
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()
        self._labels = self.df["is_moderator"].astype(float).tolist()

        # pre-compute lengths, since the text will be tokenized later
        self._lengths = []
        for txt, rep_id in zip(self._texts, self._reply_to):
            ctx = self._id2text.get(rep_id, "") if pd.notna(rep_id) else ""
            seq = f"<CONTEXT> {ctx} <COMMENT> {txt}"
            # length *after* adding special tokens, truncation etc.
            n_tokens = len(
                self.tokenizer.encode(
                    seq,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                )
            )
            self._lengths.append(n_tokens)

    def length(self, idx: int) -> int:
        return self._lengths[idx]

    # ----------------------------------------------------------------- magic
    def __len__(self):  # type: ignore[override]
        return len(self._texts)

    def __getitem__(self, idx):  # type: ignore[override]
        row = self.df.iloc[idx]
        reply_to_id = row["reply_to"]
        context = (
            self._id2text.get(reply_to_id, "") if pd.notna(reply_to_id) else ""
        )
        text = f"<CONTEXT> {context} <COMMENT> {row['text']}"
        return {"text": text, "label": float(row["is_moderator"])}


class WeightedLossTrainer(transformers.Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").view(-1)
        labels = labels.view(-1)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


class SmartBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    Yields lists of indices such that items inside a batch have similar
    sequence length.  That keeps padding - and thus compute - to a minimum.

    • Batches are *shuffled* every epoch, but order inside each batch
      is irrelevant (Trainer will not re-shuffle).
    • Works for any dataset that yields a dict with 'input_ids'.
    """

    def __init__(self, dataset, batch_size, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # just ask the dataset
        self.lengths = [dataset.length(i) for i in range(len(dataset))]
        self.sorted_indices = sorted(
            range(len(dataset)), key=self.lengths.__getitem__
        )

    def __iter__(self):
        # -- bucketed indices, then shuffle buckets --
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        random.shuffle(batches)  # different order every epoch
        for b in batches:
            yield b

    def __len__(self):
        full, rest = divmod(len(self.sorted_indices), self.batch_size)
        return full if (rest == 0 or self.drop_last) else full + 1


class BucketedTrainer(WeightedLossTrainer):
    """
    Overrides get_train_dataloader() so Trainer uses our bucketed sampler.
    """

    def __init__(self, bucket_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_batch_size = bucket_batch_size

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training dataset must be provided")

        sampler = SmartBucketBatchSampler(
            self.train_dataset,
            batch_size=self.bucket_batch_size,
            drop_last=False,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,  # provided by base Trainer
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )


class EarlyStoppingWithWarmupStepsCallback(transformers.TrainerCallback):
    def __init__(
        self,
        warmup_steps=500,
        patience=3,
        threshold=0.0,
        metric_name="eval_loss",
        greater_is_better=False,
    ):
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.wait_count = 0

    def on_evaluate(
        self,
        args,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        metrics,
        **kwargs,
    ):
        current_step = state.global_step

        if current_step < self.warmup_steps:
            return control  # still warming up

        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control  # metric not available yet

        if self.best_metric is None:
            self.best_metric = metric_value
            return control

        operator = (
            (lambda a, b: a > b + self.threshold)
            if self.greater_is_better
            else (lambda a, b: a < b - self.threshold)
        )

        if operator(metric_value, self.best_metric):
            self.best_metric = metric_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                control.should_training_stop = True

        return control


def train_model(
    model,
    tokenizer,
    train_dat,
    val_dat,
    test_dat,
    freeze_base_model: bool,
    pos_weight: float,
    output_dir: Path,
    logs_dir: Path,
):
    finetuned_model_dir = output_dir / "best_model"
    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=util.classification.BATCH_SIZE,
        per_device_eval_batch_size=util.classification.BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        eval_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        save_strategy="steps",
        save_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        logging_strategy="steps",
        logging_dir=logs_dir,
        logging_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=True,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        report_to="tensorboard",
    )

    early_stopping = EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = BucketedTrainer(
        bucket_batch_size=util.classification.BATCH_SIZE,
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=util.classification.compute_metrics,
        callbacks=[early_stopping],
        data_collator=lambda batch: collate_fn(tokenizer, batch),
    )

    checkpoints_exist = finetuned_model_dir.is_dir()
    trainer.train(resume_from_checkpoint=checkpoints_exist)

    results = trainer.evaluate(eval_dataset=test_dat)
    print(results)

    trainer.save_model(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)


def load_model_tokenizer():
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        reference_compile=False,attn_implementation="eager",
        num_labels=1,
        problem_type="multi_label_classification",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL,
    )
    return model, tokenizer


def collate_fn(tokenizer, batch: list[dict[str, str | float]]):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch]).unsqueeze(
        1
    )  # shape (B,1)

    enc = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=util.classification.MAX_LENGTH,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def main(args) -> None:
    dataset_path = Path(args.dataset_path)
    dataset_ls = args.datasets.split(",")
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    print("Starting training with datasets: ", dataset_ls)
    util.classification.set_seed(util.classification.SEED)
    model, tokenizer = load_model_tokenizer()

    print("Loading data...")
    df = pd.read_csv(dataset_path, nrows=1000)
    df = util.classification.preprocess_dataset(df, dataset_ls)
    pos_weight = (df.is_moderator == 0).sum() / (df.is_moderator == 1).sum()

    train_df, val_df, test_df = util.classification._train_validate_test_split(
        df,
        stratify_col="is_moderator",
        train_percent=0.7,
        validate_percent=0.2,
    )

    train_dataset = DiscussionModerationDataset(
        train_df, tokenizer, util.classification.MAX_LENGTH
    )
    val_dataset = DiscussionModerationDataset(
        val_df, tokenizer, util.classification.MAX_LENGTH
    )
    test_dataset = DiscussionModerationDataset(
        test_df, tokenizer, util.classification.MAX_LENGTH
    )

    print("Starting training...")
    train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        test_dataset,
        freeze_base_model=FINETUNE_ONLY_HEAD,
        pos_weight=pos_weight,
        output_dir=output_dir,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets",
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The path of the whole dataset",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
        required=True,
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Directory for training logs",
        required=True,
    )
    args = parser.parse_args()
    main(args)
