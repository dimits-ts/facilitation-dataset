import argparse
import re
from pathlib import Path

import pandas as pd

from ..util import io
from ..util import classification


SEED = 42
TOTAL_SAMPLES = 730
NUM_DECOYS = 20


def get_comments_with_context(
    full_df: pd.DataFrame, target_df: pd.DataFrame, context_len: int
) -> pd.Series:
    target_df = target_df.copy()
    full_df = full_df.copy()
    # Create a DiscussionDataset (your class handles context building)
    dataset = classification.DiscussionDataset(
        target_df=target_df,
        full_df=full_df,
        tokenizer=None,  # unused
        max_length_chars=1500,
        label_column="is_moderator",  # unused here
        max_context_turns=context_len,
    )

    # Build sequences for each row
    texts = []
    for i in range(len(dataset)):
        record = dataset[i]
        texts.append(record["text"])

    # Create a Series aligned with target_df indices
    return pd.Series(texts, index=target_df.index)


def make_human_readable(original: str) -> str:
    """
    Convert XML-like annotated chat with <CTX> ... </CTX> and <TGT> ... </TGT>
    into a clean human-readable format.
    """

    # Normalize tag variants like <\CTX> or </CTX>
    cleaned = re.sub(r"<\\?(/?)(CTX|TRT)>", r"<\1\2>", original)

    # Extract all CTX blocks
    ctxs = re.findall(r"<CTX>\s*(.*?)\s*</CTX>", cleaned, flags=re.DOTALL)

    # Extract TRT (normally one block)
    trts = re.findall(r"<TGT>\s*(.*?)\s*</TGT>", cleaned, flags=re.DOTALL)

    # Build readable output
    out_lines = []

    if ctxs:
        out_lines.append("Context:")
        for i, c in enumerate(ctxs, start=1):
            # collapse whitespace in each comment
            c = " ".join(c.split())
            out_lines.append(f"{i}. {c}")
        out_lines.append("")  # blank line

    if trts:
        out_lines.append("Target:")
        # normally one target, but support multiple
        for t in trts:
            t = " ".join(t.split())
            out_lines.append(t)

    return "\n".join(out_lines).strip()


def main(pefk_path: Path, output_dir: Path):
    df = io.progress_load_csv(pefk_path)
    df.text = df.text.astype(str)
    df.dataset = df.dataset.replace(
        {
            "wikiconv": "wikipedia",
            "wikitactics": "wikipedia",
            "wikidisputes": "wikipedia",
        }
    )
    target_df = df.loc[(df.text.str.len() >= 50) & (df.text.str.len() <= 1500)]
    target_df.text = get_comments_with_context(
        full_df=df, target_df=target_df, context_len=2
    )

    samples_per_dataset = int(TOTAL_SAMPLES / len(target_df.dataset.unique()))

    # Prepare a dictionary of per-dataset shuffled DataFrames
    shuffled = {
        dataset: target_df[target_df.dataset == dataset].reset_index(drop=True)
        for dataset in target_df.dataset.unique()
    }

    all_slices = []
    for dataset, sdf in shuffled.items():
        start = 0
        end = samples_per_dataset
        slice_df = sdf.iloc[start:end]
        all_slices.append(slice_df)

    output_df = pd.concat(all_slices, ignore_index=True)
    output_df = output_df.loc[:, ["message_id", "text"]]
    output_df.text = output_df.text.apply(make_human_readable)
    output_df = output_df.rename(columns={"message_id": "id"})

    # add consistency rows
    dup = output_df.sample(n=NUM_DECOYS, replace=False, random_state=SEED)
    output_df = pd.concat([output_df, dup], ignore_index=True)

    # shuffle final dataframe
    output_df = output_df.sample(frac=1, random_state=SEED).reset_index(
        drop=True
    )

    output_path = output_dir / "human_annotation.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Dataset exported to {output_path.resolve()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the annotation dataset"
    )
    parser.add_argument(
        "--pefk-path",
        required=True,
        help="The path to the full dataset",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="The directory for the exported annotation dataset",
    )

    args = parser.parse_args()
    main(pefk_path=Path(args.pefk_path), output_dir=Path(args.output_dir))
