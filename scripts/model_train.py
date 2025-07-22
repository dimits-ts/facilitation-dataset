from pathlib import Path
import argparse
import random

import pandas as pd
import torch
import transformers
import sklearn.metrics

import util.classification
import util.io


EVAL_STEPS = 4000
EPOCHS = 120
MAX_LENGTH = 8192
BATCH_SIZE = 24
EARLY_STOP_WARMUP = 12000
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 5
FINETUNE_ONLY_HEAD = True
TEST_METRICS = {"loss", "accuracy", "f1"}
CTX_LENGTH_COMMENTS = 2
MODEL = "answerdotai/ModernBERT-base"


class DiscussionModerationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int,
        label_column: str,
        max_context_turns: int = CTX_LENGTH_COMMENTS,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column
        self.max_context_turns = max_context_turns

        self._id2row = self.df.set_index("message_id").to_dict("index")
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()
        self._message_ids = self.df["message_id"].tolist()
        self._labels = self.df[self.label_column].astype(float).tolist()

        self._lengths = []
        for idx in range(len(self.df)):
            tokens = self._tokenized_length(idx)
            self._lengths.append(tokens)

    def _build_context(self, idx):
        """
        Build N-turn context (with usernames) for a comment at `idx`.
        """
        context_turns = []
        current_id = self.df.at[idx, "reply_to"]
        turns = 0

        while pd.notna(current_id) and turns < self.max_context_turns:
            prev = self._id2row.get(current_id)
            if not prev:
                break
            turn = f"<CTX> <USR>{prev['user']}</USR> {prev['text']} </CTX>"
            context_turns.insert(0, turn)
            current_id = prev["reply_to"]
            turns += 1

        return context_turns

    def _build_sequence(self, idx):
        context = self._build_context(idx)
        target_user = self.df.at[idx, "user"]
        target_text = self.df.at[idx, "text"]
        target = f"<TGT> <USR>{target_user}</USR> {target_text} </TGT>"
        return " ".join(context + [target])

    def _tokenized_length(self, idx):
        seq = self._build_sequence(idx)
        return len(
            self.tokenizer.encode(
                seq,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
        )

    def length(self, idx: int) -> int:
        return self._lengths[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self._build_sequence(idx)
        label = float(self.df.at[idx, self.label_column])
        return {"text": text, "label": label}


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
            self.sorted_indices[i : i + self.batch_size]
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
    Overrides get_train_dataloader() and get_eval_dataloader()
    so Trainer uses our bucketed sampler for train and default sampler
    + collate_fn for eval.
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
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    def get_eval_dataloader(
        self, eval_dataset=None
    ) -> torch.utils.data.DataLoader:
        eval_dataset = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation dataset must be provided")

        # Use default sequential sampler for eval + your collate_fn
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.bucket_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
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
    train_dat,
    val_dat,
    freeze_base_model: bool,
    pos_weight: float,
    output_dir: Path,
    logs_dir: Path,
    tokenizer,
):
    def collate(batch):
        return collate_fn(tokenizer, batch)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=1,
        problem_type="multi_label_classification",
    )

    finetuned_model_dir = output_dir / "best_model"
    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_dir=logs_dir,
        logging_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
    )

    early_stopping = EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = BucketedTrainer(
        bucket_batch_size=BATCH_SIZE,
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=util.classification.compute_metrics,
        callbacks=[early_stopping],
        data_collator=collate,
    )

    checkpoint = finetuned_model_dir if finetuned_model_dir.is_dir() else None
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)


def results_to_df(individual_raw: str, all_raw: str) -> pd.DataFrame:
    # ── parse keys →   eval_<dataset>_<metric>  ──────────────────────────────
    per_ds: dict[str, dict[str, float]] = {}
    for k, v in individual_raw.items():
        if not k.startswith("eval_"):
            continue
        _, rest = k.split("_", 1)  # drop leading 'eval_'
        # split once from the right so dataset names with '_' are handled
        ds, metric = rest.rsplit("_", 1)
        if metric in TEST_METRICS:
            per_ds.setdefault(ds, {})[metric] = v

    # ── add aggregate (“ALL”) row ────────────────────────────────────────────
    per_ds["ALL"] = {m: all_raw[f"eval_{m}"] for m in TEST_METRICS}

    # ── to DataFrame, ordered columns ────────────────────────────────────────
    df_metrics = (
        pd.DataFrame.from_dict(per_ds, orient="index")
        .reindex(columns=["loss", "accuracy", "f1"])  # fixed order
        .sort_index()
    )

    return df_metrics


def test_model(
    output_dir: Path,
    test_df: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizerBase,
    label_column: str,
) -> pd.DataFrame:
    """
    Evaluate best checkpoint on each dataset and on the full test split.
    Returns a DataFrame indexed by dataset name with columns:
    loss, accuracy, f1
    """

    # ── load checkpoint ────────────────────────────────────────────────────
    best_model_dir = output_dir / "best_model"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=1,
        problem_type="multi_label_classification",
    )

    # ── build eval datasets dict ─────────────────────────────────────────────
    def make_ds(df):
        return DiscussionModerationDataset(
            df.reset_index(drop=True),
            tokenizer,
            MAX_LENGTH,
            label_column=label_column,
        )

    eval_dict = {name: make_ds(df) for name, df in test_df.groupby("dataset")}
    full_ds = make_ds(test_df)

    trainer = BucketedTrainer(
        bucket_batch_size=BATCH_SIZE,
        pos_weight=1.0,  # not used in eval mode
        model=model,
        args=transformers.TrainingArguments(
            output_dir=output_dir / "eval",
            do_train=False,
            per_device_eval_batch_size=BATCH_SIZE,
            disable_tqdm=True,
        ),
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=util.classification.compute_metrics,
        data_collator=lambda b: collate_fn(tokenizer, b),
    )

    individual_raw = trainer.evaluate(eval_dataset=eval_dict)
    all_raw = trainer.evaluate(eval_dataset=full_ds)
    res_df = results_to_df(individual_raw=individual_raw, all_raw=all_raw)
    return res_df


def precision_recall_table(
    model_dir: Path,
    dataset: torch.utils.data.Dataset,
    tokenizer,
    thresholds: list[float],
):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=1,
        problem_type="multi_label_classification",
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(tokenizer, b),
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            inputs = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k != "labels"
            }
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu()
            all_logits.extend(torch.sigmoid(logits).numpy())
            all_labels.extend(labels.squeeze(-1).cpu().numpy())

    data = []
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in all_logits]
        precision, recall, f1, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                all_labels, preds, average="binary", zero_division=0
            )
        )
        data.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    df = pd.DataFrame(data).round(4)
    return df


def collate_fn(tokenizer, batch: list[dict[str, str | float]]):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch]).unsqueeze(1)

    enc = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def main(args) -> None:
    dataset_path = Path(args.dataset_path)
    dataset_ls = args.datasets.split(",")
    only_test = args.only_test
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    target_label = args.target_label

    print("Selected datasets: ", dataset_ls)
    util.classification.set_seed(util.classification.SEED)

    print("Loading data...")
    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls)
    pos_weight = (df[target_label] == 0).sum() / (df[target_label] == 1).sum()

    train_df, val_df, test_df = util.classification._train_validate_test_split(
        df,
        stratify_col=target_label,
        train_percent=0.7,
        validate_percent=0.2,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    train_dataset = DiscussionModerationDataset(
        train_df, tokenizer, MAX_LENGTH, target_label
    )
    val_dataset = DiscussionModerationDataset(
        val_df, tokenizer, MAX_LENGTH, target_label
    )

    if not only_test:
        print("Starting training...")

        train_model(
            train_dataset,
            val_dataset,
            tokenizer=tokenizer,
            freeze_base_model=FINETUNE_ONLY_HEAD,
            pos_weight=pos_weight,
            output_dir=output_dir,
            logs_dir=logs_dir,
        )

    print("Testing...")
    res_df = test_model(
        output_dir=output_dir,
        test_df=test_df,
        tokenizer=tokenizer,
        label_column=target_label,
    )

    print("\n=== Results ===")
    print(res_df)
    logs_path = logs_dir / "res.csv"
    res_df.to_csv(logs_path)
    print(f"Results saved to {logs_path}")

    print("\n=== PR-curves ===")
    best_model_dir = output_dir / "best_model"
    res_path = logs_dir / "pr_curves.csv"
    pr_df = precision_recall_table(
        model_dir=best_model_dir,
        dataset=val_dataset,
        tokenizer=tokenizer,
        thresholds=[
            round(t, 2) for t in list(torch.linspace(0.0, 1.0, 21).numpy())
        ],
    )
    print(pr_df)
    pr_df.to_csv(res_path, index=False)
    print(f"PR curves saved to {res_path}")


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
    parser.add_argument(
        "--only_test",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="is_moderator",
        choices=["is_moderator", "escalated"],
        help="Which column to use as the target label",
    )
    args = parser.parse_args()
    main(args)
