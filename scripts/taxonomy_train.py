from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import torch
import torch.utils.data
import transformers
import sklearn.metrics
from tqdm.auto import tqdm

import util.classification
import util.io


EVAL_STEPS = 500
EPOCHS = 150
MAX_LENGTH = 8192
BATCH_SIZE = 24
EARLY_STOP_WARMUP = 1000
EARLY_STOP_THRESHOLD = 0.001
EARLY_STOP_PATIENCE = 5
FINETUNE_ONLY_HEAD = True
MODEL = "answerdotai/ModernBERT-base"
CTX_LENGTH_COMMENTS = 4


def load_labels(base_df: pd.DataFrame, labels_dir: Path) -> pd.DataFrame:
    df = base_df.copy()
    for label_file in tqdm(
        list(labels_dir.glob("*.csv")), desc="Loading labels"
    ):
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

        df[label_name] = (
            df[label_name]
            # object -> boolean
            .infer_objects(copy=False)
            .astype("boolean")
            .fillna(False)
            .astype(int)
        )

    return df


def train_model(
    train_ds,
    val_ds,
    freeze_base_model: bool,
    output_dir: Path,
    logs_dir: Path,
    tokenizer,
    label_names: list[str],
) -> None:
    def collate(batch):
        return collate_fn(tokenizer, batch, num_labels)

    num_labels = len(label_names)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to("cuda")

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
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda eval_pred: compute_metrics_multi(
            eval_pred, label_names=label_names
        ),
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


def evaluate_model(
    model_dir: Path,
    tokenizer,
    test_ds,
    label_names: list[str],
    batch_size=24,
):
    """
    Compute per-label accuracy and F1 on test_ds.
    Returns a DataFrame with one row per label and micro/macro metrics.
    """
    # ── load checkpoint ────────────────────────────────────────────────────
    best_model_dir = model_dir / "best_model"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        best_model_dir,
        reference_compile=False,
        attn_implementation="eager",
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    model.eval()
    preds = []
    labels = []

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=lambda b: collate_fn(tokenizer, b, len(label_names)),
    )

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            batch_labels = batch["labels"].cpu().numpy()
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            batch_preds = (logits.cpu().numpy() > 0).astype(int)

            preds.append(batch_preds)
            labels.append(batch_labels)

    preds = np.vstack(preds)
    labels = np.vstack(labels)

    results = []
    for i, name in enumerate(label_names):
        results.append(
            {
                "label": name,
                "accuracy": sklearn.metrics.accuracy_score(
                    labels[:, i], preds[:, i]
                ),
                "f1": sklearn.metrics.f1_score(labels[:, i], preds[:, i]),
            }
        )

    # Add aggregate metrics
    results.append(
        {
            "label": "micro_f1",
            "accuracy": None,
            "f1": sklearn.metrics.f1_score(labels, preds, average="micro"),
        }
    )
    results.append(
        {
            "label": "macro_f1",
            "accuracy": None,
            "f1": sklearn.metrics.f1_score(labels, preds, average="macro"),
        }
    )

    return pd.DataFrame(results)


def compute_metrics_multi(eval_pred, label_names=None):
    """
    Compute per-label accuracy/F1 and aggregate micro/macro F1.

    Args:
        eval_pred: tuple of (logits, labels) from HF Trainer
        label_names: optional list of label names

    Returns:
        dict of metrics
    """
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)

    results = {}
    n_labels = labels.shape[1] if labels.ndim > 1 else 1

    for i in range(n_labels):
        name = (
            label_names[i]
            if label_names is not None and i < len(label_names)
            else f"label{i}"
        )
        results[f"accuracy_{name}"] = sklearn.metrics.accuracy_score(
            labels[:, i], preds[:, i]
        )
        results[f"f1_{name}"] = sklearn.metrics.f1_score(
            labels[:, i], preds[:, i]
        )

    # aggregate metrics
    results["micro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="micro"
    )
    results["macro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="macro"
    )

    return results


def collate_fn(tokenizer, batch: list[dict[str, str | list]], num_labels: int):
    texts = [b["text"] for b in batch]
    labels = torch.stack(
        [torch.as_tensor(b["label"], dtype=torch.float) for b in batch]
    )

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
    def make_dataset(target_df, tokenizer):
        return util.classification.DiscussionDataset(
            target_df,
            full_df=df,  # full dataset for context
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            label_column=label_names,
            max_context_turns=CTX_LENGTH_COMMENTS,
        )

    dataset_path = Path(args.dataset_path)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    label_names = [f.stem for f in labels_dir.glob("*.csv")]
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    results_csv_path = logs_dir / "res.csv"
    util.classification.set_seed(util.classification.SEED)

    # ================ Dataset preprocessing ================
    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)
    df = load_labels(df, labels_dir)

    # keep only annotated rows as targets
    target_mask = df[label_names].sum(axis=1) > 0
    target_df = df[target_mask].copy()

    train_df, val_df, test_df = util.classification.train_validate_test_split(
        target_df,
        train_percent=0.7,
        validate_percent=0.2,
    )

    # ================ Training ================
    if not args.only_test:
        print("Creating train dataset...")
        train_ds = make_dataset(train_df, tokenizer)
        print("Creating validation dataset...")
        val_ds = make_dataset(val_df, tokenizer)
        print("Starting training")
        train_model(
            train_ds,
            val_ds,
            FINETUNE_ONLY_HEAD,
            output_dir,
            logs_dir,
            tokenizer,
            label_names,
        )
        print("Training complete.")
    else:
        print("Skipping training as per cmd argument.")

    # ================ Evaluation ================
    print("Creating test dataset...")
    test_ds = make_dataset(test_df, tokenizer)
    print("Evaluating model...")
    res_df = evaluate_model(
        model_dir=output_dir,
        test_ds=test_ds,
        tokenizer=tokenizer,
        label_names=label_names,
        batch_size=BATCH_SIZE,
    )
    print("Results")
    print(res_df)
    res_df.to_csv(results_csv_path)
    print(f"Results saved to {results_csv_path}")


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
