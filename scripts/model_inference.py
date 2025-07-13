"""Infer moderator probabilities and append them to the dataset.

This variant keeps the raw sigmoid probabilities for every message
instead of thresholding them to 0/1 and selecting rows with 1s.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch
import datasets
from tqdm.auto import tqdm

import util.classification


def infer_moderator(annotated_df: pd.DataFrame, model, tokenizer):
    """Run inference and return a list with one probability per row.

    The returned list is ordered identically to ``annotated_df``.
    """
    df = annotated_df.copy()

    # Build model-ready texts with <CONTEXT> / <COMMENT> markers.
    print("Formatting dataset …")
    input_texts, _ = util.classification._build_discussion_dataset(df)
    df["input_text"] = input_texts

    # Move model to the appropriate device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Beginning inference on device", device)

    dataloader = _get_dataloader(df, tokenizer)

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.view(-1)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())

    return all_probs


def _get_dataloader(df: pd.DataFrame, tokenizer):
    """Tokenise the texts and build a simple DataLoader."""
    dataset = datasets.Dataset.from_dict({"text": df["input_text"].tolist()})
    dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=util.classification.MAX_LENGTH,
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return torch.utils.data.DataLoader(
        dataset, batch_size=util.classification.BATCH_SIZE
    )


def main(args):
    model_dir = Path(args.model_dir)
    source_dataset_path = Path(args.source_dataset_path)
    destination_dataset_path = Path(args.destination_dataset_path)
    destination_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = util.classification.load_trained_model_tokenizer(
        model_dir
    )

    print("Loading dataset for inference...")
    df = pd.read_csv(source_dataset_path)

    print("Processing dataset …")
    annotated_df = util.classification.preprocess_dataset(df)
    if annotated_df.empty:
        print("No rows match the selected datasets.")
        return

    moderator_probs = infer_moderator(annotated_df, model, tokenizer)
    annotated_df["moderator_prob"] = moderator_probs

    # Join probabilities back to the original DataFrame on ``message_id``
    df = df.merge(
        annotated_df[["message_id", "moderator_prob"]],
        on="message_id",
        how="left",
    )

    print("Exporting dataset...")
    df.to_csv(destination_dataset_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer facilitative comment probabilities"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Checkpoint directory for trained model",
    )
    parser.add_argument(
        "--source_dataset_path",
        type=str,
        required=True,
        help="Path for the source dataset",
    )
    parser.add_argument(
        "--destination_dataset_path",
        type=str,
        required=True,
        help="Path for the exported dataset",
    )
    args = parser.parse_args()

    main(args)
