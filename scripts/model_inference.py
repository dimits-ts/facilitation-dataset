"""Infer moderator probabilities and append them to the dataset.

This variant keeps the raw sigmoid probabilities for every message
instead of thresholding them to 0/1 and selecting rows with 1s.
"""

from pathlib import Path
import argparse

import pandas as pd
import torch
from tqdm.auto import tqdm

import util.classification


class _DiscussionDataset(torch.utils.data.Dataset):
    """
    Map-style dataset that, on-the-fly, turns a dataframe row into
    the combined  <CONTEXT> … <COMMENT> …  string required by the model.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        # quick look‑up: message_id → text
        self._id2text: dict[str, str] = df.set_index("message_id")[
            "text"
        ].to_dict()
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> str:
        context = ""
        r_id = self._reply_to[idx]
        if pd.notna(r_id) and r_id in self._id2text:
            context = self._id2text[r_id]
        return f"<CONTEXT> {context} <COMMENT> {self._texts[idx]}"


def _build_dataloader(
    df: pd.DataFrame, tokenizer, batch_size: float, max_length: float
) -> torch.util.DataLoader:
    """
    Returns a DataLoader that tokenises batches on-the-fly, so no giant
    tensor object is ever kept in RAM.
    """
    dataset = _DiscussionDataset(df)

    def collate_fn(batch_texts):
        return tokenizer(
            batch_texts,
            # max_length is needed for sparse attention
            # but it takes two orders of magnitude more to compute
            # so I'll just take the performance hit of full attention.
            # Most comments are short anyway
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # keep original order
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def infer_moderator(
    annotated_df: pd.DataFrame, model, tokenizer
) -> list[float]:
    """Run inference and return a list with one probability per row.

    The returned list is ordered identically to ``annotated_df``.
    """
    # df = annotated_df.copy()

    # Build model-ready texts with <CONTEXT> / <COMMENT> markers.

    # Move model to the appropriate device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Beginning inference on device", device)

    dataloader = _build_dataloader(
        annotated_df,
        tokenizer,
        batch_size=util.classification.BATCH_SIZE - 2,  # just to be sure
        max_length=util.classification.MAX_LENGTH,
    )

    all_probs = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Running inference"):
            # move *entire* batch dict to GPU in one go
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            logits = model(**batch).logits.squeeze(-1)  # shape: (B,)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())

    return all_probs


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

    print("Processing dataset...")
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
