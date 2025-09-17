from pathlib import Path
import argparse

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
MODEL = "answerdotai/ModernBERT-base"
CTX_LENGTH_COMMENTS = 4


def load_labels(base_df: pd.DataFrame, labels_dir: Path) -> pd.DataFrame:
    df = base_df.copy()
    for label_file in labels_dir.glob("*.csv"):
        label_name = label_file.stem
        label_df = pd.read_csv(label_file)
        if not {"message_id", "is_match"}.issubset(label_df.columns):
            raise ValueError(
                f"{label_file} must have 'message_id' and 'is_match'"
            )
        df = df.merge(
            label_df[["message_id", "is_match"]].rename(
                columns={"is_match": label_name}
            ),
            on="message_id",
            how="left",
        )
        df[label_name] = df[label_name].fillna(False).astype(int)
    return df


def train_model(
    train_ds,
    val_ds,
    freeze_base_model: bool,
    pos_weights,
    output_dir: Path,
    logs_dir: Path,
    tokenizer,
    num_labels: int,
):
    def collate(batch):
        return collate_fn(tokenizer, batch, num_labels)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

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

    early_stopping = util.classification.EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = util.classification.BucketedTrainer(
        bucket_batch_size=BATCH_SIZE,
        pos_weight=torch.tensor(pos_weights),
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_multi,
        callbacks=[early_stopping],
        data_collator=collate,
    )

    checkpoint = (
        (output_dir / "best_model")
        if (output_dir / "best_model").is_dir()
        else None
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")


def compute_metrics_multi(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    results = {}
    for i in range(labels.shape[1]):
        results[f"accuracy_label{i}"] = sklearn.metrics.accuracy_score(
            labels[:, i], preds[:, i]
        )
        results[f"f1_label{i}"] = sklearn.metrics.f1_score(
            labels[:, i], preds[:, i]
        )
    results["micro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="micro"
    )
    results["macro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="macro"
    )
    return results


def collate_fn(tokenizer, batch: list[dict[str, str | list]], num_labels: int):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch]).float()

    enc = tokenizer(
        texts,
        padding="longest",
        truncation=False,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def main(args):
    dataset_path = Path(args.dataset_path)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)

    util.classification.set_seed(util.classification.SEED)

    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)
    df = load_labels(df, labels_dir)

    label_names = [f.stem for f in labels_dir.glob("*.csv")]
    num_labels = len(label_names)

    pos_weights = []
    for name in label_names:
        neg, pos = (df[name] == 0).sum(), (df[name] == 1).sum()
        pos_weights.append(neg / max(pos, 1))

    train_df, val_df, test_df = util.classification._train_validate_test_split(
        df,
        train_percent=0.7,
        validate_percent=0.2,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    def make_dataset(df_):
        return util.classification.DiscussionDataset(
            df_,
            tokenizer,
            MAX_LENGTH,
            label_column=label_names,
            max_context_turns=CTX_LENGTH_COMMENTS,
        )

    # adapt dataset class to support multi-label
    train_ds = make_dataset(train_df)
    val_ds = make_dataset(val_df)

    if not args.only_test:
        train_model(
            train_ds,
            val_ds,
            FINETUNE_ONLY_HEAD,
            pos_weights,
            output_dir,
            logs_dir,
            tokenizer,
            num_labels,
        )

    print("Training complete. (Testing logic can be added similarly.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-label classification trainer"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to main dataset CSV",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Directory with per-label CSVs",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--logs_dir", type=str, required=True, help="Logs directory"
    )
    parser.add_argument(
        "--only_test", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    main(args)
