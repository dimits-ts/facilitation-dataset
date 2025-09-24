from pathlib import Path
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers

import util.classification
import util.io

MAX_LENGTH = 8192  # same as training
MODEL = "answerdotai/ModernBERT-base"  # same as training
CTX_LENGTH_COMMENTS = 2
BATCH_SIZE = 64


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

    # ── Load label names ─────────────────────────────────────
    label_names = [f.stem for f in labels_dir.glob("*.csv")]

    # ── Load dataset ─────────────────────────────────────────
    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)

    if "text" not in df.columns or "message_id" not in df.columns:
        raise ValueError(
            "Dataset must contain 'message_id' and 'text' columns."
        )

    # ── Merge label placeholders (so dataset works with DiscussionDataset) ──
    for label_name in label_names:
        if label_name not in df.columns:
            df[label_name] = 0  # dummy label for inference

    # ── Prepare DiscussionDataset ───────────────────────────
    print("Creating dataset...")
    dataset = util.classification.DiscussionDataset(
        target_df=df,
        full_df=df,  # full dataset needed for context
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        label_column=label_names,
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    device = model.device
    results = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(tokenizer, b, len(label_names)),
    )

    # ── Run inference ───────────────────────────────────────
    for batch in tqdm(loader, desc="Running inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        preds = (logits > 0).int().cpu().numpy()

        batch_message_ids = [
            dataset._message_ids[i] for i in range(len(batch["input_ids"]))
        ]
        for mid, pred in zip(batch_message_ids, preds):
            row = {"message_id": mid}
            for label_name, p in zip(label_names, pred):
                row[label_name] = int(p)
            results.append(row)

    # ── Save predictions ─────────────────────────────────────
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
