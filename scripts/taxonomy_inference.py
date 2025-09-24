from pathlib import Path
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers

import util.classification
import util.io

MAX_LENGTH = 8192
BATCH_SIZE = 32
MODEL = "answerdotai/ModernBERT-base"  # same as training


def collate_fn(tokenizer, batch, max_length):
    texts = [b["text"] for b in batch]
    enc = tokenizer(
        texts,
        padding="longest",
        truncation=False,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc


@torch.no_grad()
def run_inference(
    model_dir: Path,
    dataset_path: Path,
    labels_dir: Path,
    output_csv: Path,
):
    # ── Load tokenizer and model ──────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_dir / "best_model"
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir / "best_model",
        reference_compile=False,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    # ── Load label names (from the training label CSVs) ───────
    label_names = [f.stem for f in labels_dir.glob("*.csv")]
    print(f"Detected labels: {label_names}")

    # ── Load dataset ──────────────────────────────────────────
    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)

    if "text" not in df.columns or "message_id" not in df.columns:
        raise ValueError(
            "Dataset must contain 'message_id' and 'text' columns."
        )

    device = model.device

    # ── Prepare batches ───────────────────────────────────────
    results = []
    model.eval()

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Running inference"):
        batch_df = df.iloc[i: i + BATCH_SIZE]
        batch = [{"text": text} for text in batch_df["text"].tolist()]

        enc = collate_fn(tokenizer, batch, MAX_LENGTH)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        preds = (logits > 0).int().cpu().numpy()

        for mid, pred in zip(batch_df["message_id"], preds):
            row = {"message_id": mid}
            for label_name, p in zip(label_names, pred):
                row[label_name] = int(p)
            results.append(row)

    # ── Save predictions ──────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with trained multi-label classifier"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input dataset CSV",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Directory with label CSVs (to get label names)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save predictions CSV",
    )

    args = parser.parse_args()

    run_inference(
        model_dir=Path(args.model_dir),
        dataset_path=Path(args.dataset_path),
        labels_dir=Path(args.labels_dir),
        output_csv=Path(args.output_csv),
    )
