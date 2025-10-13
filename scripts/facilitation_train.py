from pathlib import Path
import argparse

import pandas as pd
import torch
import transformers
import sklearn.metrics

import util.classification
import util.io


EVAL_STEPS = 3000
EPOCHS = 200
MAX_LENGTH = 4096
BATCH_SIZE = 42
EARLY_STOP_WARMUP = 0
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 8
FINETUNE_ONLY_HEAD = True
TEST_METRICS = {"loss", "accuracy", "f1"}
CTX_LENGTH_COMMENTS = 4
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
    output_dir: Path,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
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
        return util.classification.DiscussionDataset(
            full_df=full_df.reset_index(drop=True),
            target_df=df.reset_index(drop=True),
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            label_column=label_column,
            max_context_turns=CTX_LENGTH_COMMENTS,
        )

    eval_dict = {name: make_ds(df) for name, df in test_df.groupby("dataset")}
    full_ds = util.classification.DiscussionDataset(
        full_df=full_df.reset_index(drop=True),
        target_df=test_df.reset_index(drop=True),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        label_column=label_column,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )
    trainer = util.classification.BucketedTrainer(
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
        data_collator=lambda b: collate_fn(tokenizer, b)
    )

    individual_raw = trainer.evaluate(eval_dataset=eval_dict)
    all_raw = trainer.evaluate(eval_dataset=full_ds)
    res_df = util.classification.results_to_df(
        individual_raw=individual_raw,
        all_raw=all_raw,
        test_metrics=TEST_METRICS,
    )
    return res_df


def precision_recall_table(
    model_dir: Path,
    dataset: torch.utils.data.Dataset,
    tokenizer,
    thresholds: list[float],
):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir,
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
        truncation=False,
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
        max_length=MAX_LENGTH,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )
    val_dataset = util.classification.DiscussionDataset(
        full_df=df,
        target_df=val_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        label_column=target_label,
        max_context_turns=CTX_LENGTH_COMMENTS,
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
        full_df=df,
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
        choices=["is_moderator", "should_intervene"],
        help="Which column to use as the target label",
    )
    args = parser.parse_args()
    main(args)
