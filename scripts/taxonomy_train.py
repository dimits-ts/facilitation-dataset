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
import util.preprocessing
import util.io


EVAL_STEPS = 2000
EPOCHS = 3500
MAX_LENGTH = 8192
BATCH_SIZE = 32
EARLY_STOP_WARMUP = 0
EARLY_STOP_THRESHOLD = 0.0001
EARLY_STOP_PATIENCE = 6
FINETUNE_ONLY_HEAD = True
MODEL = "answerdotai/ModernBERT-base"
CTX_LENGTH_COMMENTS = 2


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
        for batch in tqdm(loader, desc="Running evaluation"):
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
                "precision": sklearn.metrics.precision_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
                "recall": sklearn.metrics.recall_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
                "f1": sklearn.metrics.f1_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
            }
        )

    # Add aggregate metrics
    results.append(
        {
            "label": "micro_avg",
            "accuracy": 1 - sklearn.metrics.hamming_loss(labels, preds),
            "precision": sklearn.metrics.precision_score(
                labels, preds, average="micro", zero_division=0
            ),
            "recall": sklearn.metrics.recall_score(
                labels, preds, average="micro", zero_division=0
            ),
            "f1": sklearn.metrics.f1_score(
                labels, preds, average="micro", zero_division=0
            ),
        }
    )
    results.append(
        {
            "label": "macro_avg",
            "accuracy": 1 - sklearn.metrics.hamming_loss(labels, preds),
            "precision": sklearn.metrics.precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall": sklearn.metrics.recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            "f1": sklearn.metrics.f1_score(
                labels, preds, average="macro", zero_division=0
            ),
        }
    )

    return pd.DataFrame(results)


def compute_metrics_multi(eval_pred, label_names=None):
    """
    Compute per-label accuracy/F1/precision/recall and aggregate micro/macro.

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
        results[f"precision_{name}"] = sklearn.metrics.precision_score(
            labels[:, i], preds[:, i], zero_division=0
        )
        results[f"recall_{name}"] = sklearn.metrics.recall_score(
            labels[:, i], preds[:, i], zero_division=0
        )
        results[f"f1_{name}"] = sklearn.metrics.f1_score(
            labels[:, i], preds[:, i], zero_division=0
        )

    # aggregate metrics
    results["micro_precision"] = sklearn.metrics.precision_score(
        labels, preds, average="micro", zero_division=0
    )
    results["micro_recall"] = sklearn.metrics.recall_score(
        labels, preds, average="micro", zero_division=0
    )
    results["micro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="micro", zero_division=0
    )

    results["macro_precision"] = sklearn.metrics.precision_score(
        labels, preds, average="macro", zero_division=0
    )
    results["macro_recall"] = sklearn.metrics.recall_score(
        labels, preds, average="macro", zero_division=0
    )
    results["macro_f1"] = sklearn.metrics.f1_score(
        labels, preds, average="macro", zero_division=0
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


def get_fora_df(pefk_df: pd.DataFrame) -> pd.DataFrame:
    fora_df = util.preprocessing.get_human_df(pefk_df, "fora")
    fora_df = fora_df.drop(
        columns=["fora.Personal story", "fora.Personal experience"]
    )
    return fora_df


def get_whow_df(pefk_df: pd.DataFrame) -> pd.DataFrame:
    return util.preprocessing.get_human_df(pefk_df, "whow")


def get_label_columns(df: pd.DataFrame) -> list[str]:
    return [x for x in df.columns if "." in x]


def train_pipeline(
    target_df: pd.DataFrame,
    full_df: pd.DataFrame,
    tokenizer,
    output_dir: Path,
    logs_dir: Path,
) -> None:
    label_names = get_label_columns(target_df)
    print("Target label names: ", label_names)

    train_df, val_df, _ = util.classification.train_validate_test_split(
        target_df,
        train_percent=0.7,
        validate_percent=0.2,
    )
    print("Creating train dataset...")
    train_ds = make_dataset(train_df, full_df, tokenizer, label_names)
    print("Creating validation dataset...")
    val_ds = make_dataset(val_df, full_df, tokenizer, label_names)

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


def evaluation_pipeline(
    target_df: pd.DataFrame,
    full_df: pd.DataFrame,
    tokenizer,
    model_dir: Path,
    output_csv_path: Path,
):
    label_names = get_label_columns(target_df)
    _, _, test_df = util.classification.train_validate_test_split(
        target_df,
        train_percent=0.7,
        validate_percent=0.2,
    )
    print("Creating test dataset...")
    test_ds = make_dataset(test_df, full_df, tokenizer, label_names)
    print("Evaluating model...")
    res_df = evaluate_model(
        model_dir=model_dir,
        test_ds=test_ds,
        tokenizer=tokenizer,
        label_names=label_names,
        batch_size=BATCH_SIZE,
    )
    print("Results")
    print(res_df)
    res_df.to_csv(output_csv_path)
    print(f"Results saved to {output_csv_path}.")


def make_dataset(target_df, full_df, tokenizer, label_names):
    return util.classification.DiscussionDataset(
        target_df=target_df,
        full_df=full_df,  # full dataset for context
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        label_column=label_names,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )


def main(args):
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    util.classification.set_seed(util.classification.SEED)

    # ================ Dataset preprocessing ================
    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)
    mod_df = df[df.is_moderator == 1]
    print(
        f"Selected {len(mod_df)} mod comments for training "
        f"out of {len(df)} total comments."
    )
    fora_df = get_fora_df(mod_df)
    whow_df = get_whow_df(mod_df)

    # ================ Training ================
    if not args.only_test:
        print("Starting training for Fora...")
        train_pipeline(
            target_df=fora_df,
            full_df=df,
            tokenizer=tokenizer,
            output_dir=output_dir / "fora",
            logs_dir=logs_dir / "fora",
        )

        print("Starting training for WHoW...")
        train_pipeline(
            target_df=whow_df,
            full_df=df,
            tokenizer=tokenizer,
            output_dir=output_dir / "whow",
            logs_dir=logs_dir / "whow",
        )
    else:
        print("Skipping training as per cmd argument.")

    print("Starting evaluation for Fora...")
    evaluation_pipeline(
        target_df=fora_df,
        full_df=df,
        tokenizer=tokenizer,
        model_dir=output_dir / "fora",
        output_csv_path=logs_dir / "fora" / "res.csv",
    )
    print("Starting evaluation for WHoW...")
    evaluation_pipeline(
        target_df=whow_df,
        full_df=df,
        tokenizer=tokenizer,
        model_dir=output_dir / "whow",
        output_csv_path=logs_dir / "whow" / "res.csv",
    )

    print("Finished model pipeline.")

    # ================ Evaluation ================


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
