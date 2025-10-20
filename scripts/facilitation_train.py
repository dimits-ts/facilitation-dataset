from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
import transformers
import sklearn.metrics
from tqdm.auto import tqdm

import util.classification
import util.io


EPOCHS = 80
MAX_LENGTH_CHARS = 5000
BATCH_SIZE = 64
EARLY_STOP_WARMUP = 4
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 6
FINETUNE_ONLY_HEAD = True
CTX_LENGTH_COMMENTS = 2
MODEL = "answerdotai/ModernBERT-large"


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
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=logs_dir,
    )

    early_stopping = util.classification.EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = util.classification.BucketedTrainer(
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


def test_model(
    model,
    tokenizer: transformers.PreTrainedTokenizerBase,
    output_dir: Path,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """
    Single-pass evaluation: runs inference once on the full test set,
    computes per-dataset and overall metrics.
    """

    best_model_dir = output_dir / "best_model"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        num_labels=1,
        problem_type="multi_label_classification",
    )

    # Full test dataset (single pass)
    full_ds = util.classification.DiscussionDataset(
        full_df=full_df.reset_index(drop=True),
        target_df=test_df.reset_index(drop=True),
        tokenizer=tokenizer,
        max_length_chars=MAX_LENGTH_CHARS,
        label_column=label_column,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    logits, labels = _collect_logits_and_labels(model, full_ds, tokenizer)

    return logits, labels


def precision_recall_table_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    data = []
    for t in thresholds:
        preds = (logits >= t).astype(int)
        precision, recall, f1, _ = (
            sklearn.metrics.precision_recall_fscore_support(
                labels, preds, average="binary", zero_division=0
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
    return pd.DataFrame(data).round(4)


def collate_fn(tokenizer, batch: list[dict[str, str | float]]):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch]).unsqueeze(1)

    enc = tokenizer(
        texts,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def res_df_from_logits_and_labels(
    test_df, logits, labels, label_column: str
) -> pd.DataFrame:
    # attach predictions
    df_eval = test_df.copy()
    df_eval["logit"] = logits
    df_eval["pred"] = (df_eval["logit"] >= 0.5).astype(int)

    # compute per-dataset statistics
    rows = []
    for name, group in df_eval.groupby("dataset"):
        precision, recall, f1, support = (
            sklearn.metrics.precision_recall_fscore_support(
                group[label_column],
                group["pred"],
                average="binary",
            )
        )

        rows.append(
            {
                "dataset": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    return pd.DataFrame(rows).set_index("dataset")


def _collect_logits_and_labels(model, dataset, tokenizer):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(tokenizer, b),
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            labels = batch["labels"].squeeze(-1).cpu()
            inputs = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k != "labels"
            }
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu()
            all_logits.append(torch.sigmoid(logits))
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return logits, labels


def main(args) -> None:
    dataset_path = Path(args.dataset_path)
    dataset_ls = args.datasets.split(",")
    only_test = args.only_test
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    target_label = args.target_label

    print("Selected datasets: ", dataset_ls)
    util.classification.set_seed(util.classification.SEED)

    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls)
    # remove comment if should_intervene is the target
    # (only case where NaNs should exist)
    df = df.dropna(subset=target_label)

    pos_weight = (df[target_label] == 0).sum() / (df[target_label] == 1).sum()

    train_df, val_df, test_df = util.classification.train_validate_test_split(
        df,
        stratify_col=target_label,
        train_percent=0.8,
        validate_percent=0.1,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)

    train_dataset = util.classification.DiscussionDataset(
        full_df=df,
        target_df=train_df,
        tokenizer=tokenizer,
        max_length_chars=MAX_LENGTH_CHARS,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )
    val_dataset = util.classification.DiscussionDataset(
        full_df=df,
        target_df=val_df,
        tokenizer=tokenizer,
        max_length_chars=MAX_LENGTH_CHARS,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    print("DEBUG: Checking a few samples:")
    for i in range(5):
        ex = train_dataset[i]
        print(f"\n--- Sample {i} ---")
        print(ex["text"])
        print(f"Label: {ex['label']}\n")

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
    best_model_dir = output_dir / "best_model"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        best_model_dir
    )
    logits, labels = test_model(
        model=model,
        output_dir=output_dir,
        full_df=df,
        test_df=test_df,
        tokenizer=tokenizer,
        label_column=target_label,
    )

    print("\n=== Results ===")

    res_df = res_df_from_logits_and_labels(
        test_df=test_df,
        logits=logits,
        labels=labels,
        label_column=target_label,
    )
    print(res_df)
    res_path = logs_dir / "res_dataset.csv"
    res_df.to_csv(res_path, index=False)
    print(f"Results per dataset saved to {res_path}.")

    # run pr curves on validation set since their results are used
    # as parameters for the next stages
    logits, labels = test_model(
        model=model,
        output_dir=output_dir,
        full_df=df,
        test_df=val_df,
        tokenizer=tokenizer,
        label_column=target_label,
    )
    pr_path = logs_dir / "pr_curves.csv"
    pr_df = precision_recall_table_from_logits(
        logits,
        labels,
        thresholds=[
            round(t, 2) for t in list(torch.linspace(0.0, 1.0, 21).numpy())
        ],
    )
    print(pr_df)
    pr_path = logs_dir / "pr_curves.csv"
    pr_df.to_csv(pr_path, index=False)
    print(f"PR curves saved to {pr_path}.")


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
        choices=["is_moderator", "should_intervene"],
        help="Which column to use as the target label",
    )
    args = parser.parse_args()
    main(args)
