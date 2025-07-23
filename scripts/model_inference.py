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

import torch
import transformers
from tqdm.auto import tqdm

import util.classification
import util.io

BATCH_SIZE = 4
MAX_LENGTH = 8192
CTX_LENGTH_COMMENTS = 4


def sort_dataset_by_tokenized_sequence_length(dataset):
    lengths = [dataset.length(i) for i in range(len(dataset))]
    dataset.df["sequence_length"] = lengths
    dataset.df = dataset.df.sort_values(
        by="sequence_length", ascending=False
    ).drop(columns="sequence_length")
    return dataset


def _build_dataloader(dataset, tokenizer) -> torch.utils.data.DataLoader:
    def collate_fn(batch_items: list[dict]):
        texts = [item["text"] for item in batch_items]
        return tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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
    dataset,
    model,
    tokenizer,
    destination_dataset_path: Path,
    output_column_name: str,
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataloader = _build_dataloader(dataset, tokenizer)

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
            df_batch = dataset.df.iloc[offset:offset + len(probs)].copy()
            df_batch[output_column_name] = probs
            df_batch = df_batch.loc[:, ["message_id", output_column_name]]
            write_queue.put(df_batch)
            offset += len(probs)

    write_queue.put(None)
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
    df = util.classification.preprocess_dataset(df)

    if df.empty:
        print("No rows match filters - nothing to do.")
        return

    print("Creating dataset...")
    dataset = util.classification.DiscussionDataset(
        df,
        tokenizer,
        MAX_LENGTH,
        label_column="dummy_col",
        max_context_turns=CTX_LENGTH_COMMENTS,
    )

    print("Sorting data by tokenized sequence length...")
    dataset = sort_dataset_by_tokenized_sequence_length(dataset)

    # Inference
    infer_and_append(
        dataset=dataset,
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
