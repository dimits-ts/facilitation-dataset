#!/usr/bin/env python3
"""
Infer moderator probabilities and append **full dataframe batches** to disk.

After each mini-batch the script
    1. computes fresh probabilities,
    2. inserts them into the corresponding slice of the in-memory DataFrame, 
    3. appends **that entire slice** (all original columns + the new
       ``moderator_prob`` column) to ``--destination_dataset_path``.

This preserves crash-resumability with minimal RAM use (no long lists of logits
kept in memory) while letting downstream processes stream-read the growing CSV.
"""
from pathlib import Path
import argparse
import threading
import queue

import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

import util.classification

BATCH_SIZE = 24
MAX_LENGTH = 512


# ───────────────────────────────────── Dataset & DataLoader ──────────────────
class _DiscussionDataset(torch.utils.data.Dataset):
    """Map-style dataset turning a dataframe row into the combined
    ``<CONTEXT> … <COMMENT> …`` string required by the model (done on-the-fly).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # id -> text lookup for fast context fetch
        self._id2text = self.df.set_index("message_id")["text"].to_dict()
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._texts)

    def __getitem__(self, idx: int) -> str:  # type: ignore[override]
        context = ""
        r_id = self._reply_to[idx]
        if pd.notna(r_id) and r_id in self._id2text:
            context = self._id2text[r_id]
        return f"<CONTEXT> {context} <COMMENT> {self._texts[idx]}"


def _build_dataloader(
    df: pd.DataFrame, tokenizer
) -> torch.utils.data.DataLoader:
    """Tokenise batches on-the-fly; keeps memory footprint tiny."""

    dataset = _DiscussionDataset(df)

    def collate_fn(batch_texts: list[str]):
        return tokenizer(
            batch_texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,  # a little headroom
        shuffle=False,  # preserve order
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )


# ─────────────────────────────────────── Helpers ─────────────────────────────


def load_trained_model_tokenizer(model_dir: Path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir, reference_compile=False, attn_implementation="eager"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def _append_batch_to_csv(
    df_batch: pd.DataFrame, out_path: Path, *, first_batch: bool
) -> None:
    """Append a dataframe slice to ``out_path``.

    Header is written only on the first batch so that the final file is a valid
    CSV concatenation of all batches.
    """
    mode = "w" if first_batch else "a"
    df_batch.to_csv(out_path, mode=mode, header=first_batch, index=False)


def writer_thread_func(write_queue: queue.Queue, out_path: Path):
    """Continuously writes batches to disk from the queue."""
    first_batch = True
    while True:
        df_batch = write_queue.get()
        if df_batch is None:  # Sentinel to shut down
            break
        _append_batch_to_csv(df_batch, out_path, first_batch=first_batch)
        first_batch = False
        write_queue.task_done()


# ───────────────────────────────────── Inference loop ────────────────────────


def _infer(model, device: str, batch):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    probs = torch.sigmoid(model(**batch).logits.squeeze(-1)).cpu().tolist()
    return probs


# offload IO writing to async thread
def infer_and_append(
    annotated_df: pd.DataFrame,
    model,
    tokenizer,
    destination_dataset_path: Path,
) -> None:
    if destination_dataset_path.exists():
        print(f"{destination_dataset_path} already exists. Exiting ...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataloader = _build_dataloader(annotated_df, tokenizer)

    write_queue = queue.Queue(maxsize=8)
    writer_thread = threading.Thread(
        target=writer_thread_func,
        args=(write_queue, destination_dataset_path),
        daemon=True,
    )
    writer_thread.start()

    with torch.inference_mode():
        offset = 0
        for batch in tqdm(dataloader, desc="Running inference", leave=False):
            probs = _infer(model, device, batch)
            df_batch = annotated_df.iloc[offset:offset + len(probs)].copy()
            df_batch["moderator_prob"] = probs

            write_queue.put(df_batch)
            offset += len(probs)

    write_queue.put(None)  # Sentinel to signal writer to finish
    writer_thread.join()


# ────────────────────────────────────────── Main ─────────────────────────────


def main(args: argparse.Namespace) -> None:
    # paths ────────────────────────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    src_path = Path(args.source_dataset_path)
    dst_path = Path(args.destination_dataset_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # model ────────────────────────────────────────────────────────────────
    model, tokenizer = load_trained_model_tokenizer(model_dir)

    # load & clean ─────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(src_path)
    df = df.sort_values(by="text", key=lambda c: c.str.len(), ascending=False)
    annotated_df = util.classification.preprocess_dataset(df)
    if annotated_df.empty:
        print("No rows match filters - nothing to do.")
        return

    # inference ────────────────────────────────────────────────────────────
    infer_and_append(
        annotated_df=annotated_df,
        model=model,
        tokenizer=tokenizer,
        destination_dataset_path=dst_path,
    )

    print("Dataset written to", dst_path)


# ────────────────────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer facilitative comment probabilities and stream the "
        "full dataframe batches to disk."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Checkpoint directory for trained model.",
    )
    parser.add_argument(
        "--source_dataset_path",
        type=str,
        required=True,
        help="Path to the source dataset (CSV).",
    )
    parser.add_argument(
        "--destination_dataset_path",
        type=str,
        required=True,
        help="Path for the continuously growing output dataset (CSV).",
    )

    main(parser.parse_args())
