from pathlib import Path
import argparse

import pandas as pd
import torch
from transformers import TextClassificationPipeline
from tqdm.auto import tqdm

import util.classification


DATASET_PATH = Path("pefk.csv")


def main(args):
    dataset_ls = [ds.strip() for ds in args.datasets.split(",")]
    model_dir = args.model_dir

    df = pd.read_csv(DATASET_PATH)
    df.text = df.text.astype(str)
    df["is_moderator_inferred"] = df["dataset"].isin(dataset_ls)

    annotated_df = df[df["dataset"].isin(dataset_ls)]
    # TODO: delete
    annotated_df = annotated_df.head(1000)

    if annotated_df.empty:
        print("No rows match the selected datasets.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = util.classification.load_trained_model_tokenizer(
        model_dir
    )
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=device,
    )

    preds = []
    for i in tqdm(
        range(0, len(annotated_df), util.classification.BATCH_SIZE),
        desc="Running inference",
    ):
        batch = annotated_df.iloc[i : i + util.classification.BATCH_SIZE]
        texts = batch["text"].tolist()
        outputs = pipe(texts)
        batch_preds = [
            1 if output["label"] == "LABEL_1" else 0 for output in outputs
        ]
        preds.extend(batch_preds)

    df.loc[annotated_df.index, "is_moderator_inferred"] = preds
    # TODO: delete
    #df.to_csv(DATASET_PATH, index=False)
    df.to_csv("test.csv", index=False)


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
