"""
Infer moderator probabilities and append them to the dataset.

After each mini-batch the script
    1. writes the fresh probabilities into the in-memory DataFrame, and
    2. overwrites --destination_dataset_path with the updated DataFrame.

That gives a crash-resumable workflow with minimal RAM use
(no long lists of logits kept in memory).
"""

from pathlib import Path
import argparse

import pandas as pd
import torch
from tqdm.auto import tqdm

import util.classification as uc


# ───────────────────────────────────── Dataset & DataLoader ─────────────────
class _DiscussionDataset(torch.utils.data.Dataset):
    """
    Map-style dataset that, on-the-fly, turns a dataframe row into the
    combined  <CONTEXT> ... <COMMENT> ...  string required by the model.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # message_id → text lookup for fast context fetch
        self._id2text = self.df.set_index("message_id")["text"].to_dict()
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()

    def __len__(self):
        return len(self._texts)

    def __getitem__(self, idx: int) -> str:
        context = ""
        r_id = self._reply_to[idx]
        if pd.notna(r_id) and r_id in self._id2text:
            context = self._id2text[r_id]
        return f"<CONTEXT> {context} <COMMENT> {self._texts[idx]}"


def _build_dataloader(df: pd.DataFrame, tokenizer):
    """Tokenise batches on-the-fly; keeps memory footprint tiny."""
    dataset = _DiscussionDataset(df)

    def collate_fn(batch_texts):
        return tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=uc.MAX_LENGTH,
            return_tensors="pt",
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=uc.BATCH_SIZE - 2,  # a little headroom
        shuffle=False,  # preserve order
        collate_fn=collate_fn,
        num_workers=6,
        pin_memory=torch.cuda.is_available(),
    )


# ───────────────────────────────────── Inference loop ───────────────────────
def infer_and_stream(
    annotated_df: pd.DataFrame,
    df_full: pd.DataFrame,
    model,
    tokenizer,
    *,
    dest_path: Path,
):
    """
    Streams probabilities into `df_full` *and* writes the whole DataFrame
    to `dest_path` after every batch.

    df_full must contain a column called 'moderator_prob' (initialised to NaN).
    """

    # -- build a fast row-lookup table:  message_id → row-number
    idx_map = pd.Series(df_full.index, index=df_full["message_id"]).to_dict()

    # ordered ids that correspond to the DataLoader batches
    ordered_msg_ids = annotated_df["message_id"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("Starting inference on", device)

    dataloader = _build_dataloader(annotated_df, tokenizer)

    with torch.inference_mode():
        offset = 0
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Running inference")
        ):
            # --- model forward ------------------------------------------------
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            logits = model(**batch).logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().tolist()

            # --- write probs into df_full ------------------------------------
            ids_this_batch = ordered_msg_ids[offset:offset + len(probs)]
            row_idx = [idx_map[mid] for mid in ids_this_batch]
            df_full.loc[row_idx, "moderator_prob"] = probs
            offset += len(probs)

            # --- save entire DataFrame after this batch ----------------------
            df_full.to_csv(dest_path, index=False, )

    print("✓ Inference complete - final dataset written to", dest_path)


# ───────────────────────────────────────── Main ─────────────────────────────
def main(args):
    model_dir = Path(args.model_dir)
    src_path = Path(args.source_dataset_path)
    dst_path = Path(args.destination_dataset_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = uc.load_trained_model_tokenizer(model_dir)

    print("Loading dataset for inference...")
    df = pd.read_csv(src_path)

    # Optional length-based sort for padding efficiency
    df = df.sort_values(
        by="text", key=lambda col: col.str.len(), ascending=False
    )

    # Prepare a column to hold the probabilities
    df["moderator_prob"] = pd.NA

    print("Filtering & cleaning...")
    annotated_df = uc.preprocess_dataset(df)
    if annotated_df.empty:
        print("No rows match the selected datasets.")
        return

    infer_and_stream(
        annotated_df,
        df_full=df,
        model=model,
        tokenizer=tokenizer,
        dest_path=dst_path,
    )


# ───────────────────────────────────────── CLI ──────────────────────────────
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
        help="Path to the source dataset",
    )
    parser.add_argument(
        "--destination_dataset_path",
        type=str,
        required=True,
        help="Path for the continuously updated dataset",
    )
    main(parser.parse_args())
