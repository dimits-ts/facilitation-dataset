from pathlib import Path
import argparse

import pandas as pd
import torch
import datasets
from tqdm.auto import tqdm

import util.classification


DATASET_PATH = Path("pefk.csv")


def infer_moderator(annotated_df, model, tokenizer):
    df = annotated_df.copy()
    # format input texts with <CONTEXT> and <COMMENT> tokens
    input_texts, _ = util.classification._build_discussion_dataset(df)
    df["input_text"] = input_texts

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = _get_dataloader(df, tokenizer)

    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            batch_preds = (probs > 0.5).long().cpu().tolist()
            preds.extend(batch_preds)
    return preds


def _get_dataloader(df, tokenizer):
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

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=util.classification.BATCH_SIZE
    )
    return dataloader


def main(args):
    dataset_ls = [ds.strip() for ds in args.datasets.split(",")]
    model_dir = Path(args.model_dir)

    model, tokenizer = util.classification.load_trained_model_tokenizer(
        model_dir
    )

    df = pd.read_csv(DATASET_PATH)
    annotated_df = util.classification.preprocess_dataset(df, dataset_ls)
    if annotated_df.empty:
        print("No rows match the selected datasets.")
        return

    preds = infer_moderator(annotated_df, model, tokenizer)
    # Save predictions back to df
    annotated_df["is_moderator_inferred"] = preds
    inferred_ids = set(
        annotated_df.loc[
            annotated_df["is_moderator_inferred"] == 1, "message_id"
        ]
    )
    df["is_moderator_inferred"] = (
        df["message_id"].isin(inferred_ids).astype(bool)
    )

    df.to_csv("pefk2.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "datasets",
        type=str,
        help="Comma-separated list of datasets to be annotated",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Checkpoint directory for trained model",
        required=True,
    )
    args = parser.parse_args()
    main(args)
