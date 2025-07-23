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
import util.io

BATCH_SIZE = 4
MAX_LENGTH = 8192
CTX_LENGTH_COMMENTS = 4


def sort_by_tokenized_sequence_length(
    df, tokenizer, label_column="label", max_context_turns=5
):
    dataset = util.classification.DiscussionDataset(
        df, tokenizer, MAX_LENGTH, label_column, max_context_turns
    )
    lengths = [dataset.length(i) for i in range(len(dataset))]
    df = df.copy()
    df["sequence_length"] = lengths
    return df.sort_values(by="sequence_length", ascending=False).drop(
        columns="sequence_length"
    )


def _build_dataloader(
    df: pd.DataFrame, tokenizer
) -> torch.utils.data.DataLoader:
    """Tokenise batches on-the-fly; keeps memory footprint tiny."""
    dataset = util.classification.DiscussionDataset(
        df,
        tokenizer,
        MAX_LENGTH,
        "dummy_col",
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    def collate_fn(batch_texts: list[str]):
        return tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
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
    output_column_name: str,
) -> None:
    if destination_dataset_path.exists():
        print(f"{destination_dataset_path} already exists. Exiting ...")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataloader = _build_dataloader(annotated_df, tokenizer)

    write_queue = queue.Queue(maxsize=8)
    writer_thread = threading.Thread(
        target=util.io.writer_thread_func,
        args=(write_queue, destination_dataset_path),
        daemon=True,
    )
    writer_thread.start()

    with torch.inference_mode():
        offset = 0
        for batch in tqdm(dataloader, desc="Running inference", leave=False):
            probs = _infer(model, device, batch)
            df_batch = annotated_df.iloc[offset:offset + len(probs)].copy()
            df_batch[output_column_name] = probs
            df_batch = df_batch.loc[:, ["message_id", output_column_name]]

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
    output_column_name = args.output_column_name

    # model ────────────────────────────────────────────────────────────────
    model, tokenizer = load_trained_model_tokenizer(model_dir)

    # load & clean ─────────────────────────────────────────────────────────
    df = util.io.progress_load_csv(src_path)
    df["dummy_col"] = 0
    print("Sorting data by sequence length...")
    df = sort_by_tokenized_sequence_length(
        df, tokenizer, "dummy_col", CTX_LENGTH_COMMENTS
    )
    print("Creating dataaset...")
    annotated_df = util.classification.preprocess_dataset(df)
    annotated_df = annotated_df.loc[
        :, ["message_id", "text", "user", "reply_to"]
    ]
    if annotated_df.empty:
        print("No rows match filters - nothing to do.")
        return

    # inference ────────────────────────────────────────────────────────────
    infer_and_append(
        annotated_df=annotated_df,
        model=model,
        tokenizer=tokenizer,
        destination_dataset_path=dst_path,
        output_column_name=output_column_name,
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
    parser.add_argument(
        "--output_column_name",
        type=str,
        required=True,
        choices=["mod_probabilities", "escalation_probabilities"],
        help="How to name the new inferred column",
    )

    main(parser.parse_args())
