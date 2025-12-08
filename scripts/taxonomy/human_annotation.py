import argparse
import re
from pathlib import Path

import pandas as pd

from ..util import io
from ..util import classification


SEED = 42
TOTAL_SAMPLES = 830
NUM_DECOYS = 20


def get_comments_with_context(
    full_df: pd.DataFrame, target_df: pd.DataFrame, context_len: int
) -> pd.Series:
    target_df = target_df.copy()
    full_df = full_df.copy()

    dataset = classification.DiscussionDataset(
        target_df=target_df,
        full_df=full_df,
        tokenizer=None,  # unused
        max_length_chars=1500,
        label_column="is_moderator",  # unused
        max_context_turns=context_len,
    )

    texts = []
    for i in range(len(dataset)):
        texts.append(dataset[i]["text"])

    return pd.Series(texts, index=target_df.index)


def make_human_readable(original: str) -> str:
    cleaned = re.sub(r"<\\?(/?)(CTX|TRT)>", r"<\1\2>", original)

    ctxs = re.findall(r"<CTX>\s*(.*?)\s*</CTX>", cleaned, flags=re.DOTALL)
    trts = re.findall(r"<TGT>\s*(.*?)\s*</TGT>", cleaned, flags=re.DOTALL)

    out_lines = []

    if ctxs:
        out_lines.append("Context:")
        for i, c in enumerate(ctxs, start=1):
            c = " ".join(c.split())
            out_lines.append(f"{i}. {c}")
        out_lines.append("")

    if trts:
        out_lines.append("Target:")
        for t in trts:
            t = " ".join(t.split())
            out_lines.append(t)

    return "\n".join(out_lines).strip()


def build_dataset(df: pd.DataFrame, name: str, output_dir: Path):
    """
    Build one of the two disjoint datasets and export it.
    """

    # How many datasets
    datasets = df.dataset.unique()
    samples_per_dataset = TOTAL_SAMPLES // len(datasets)

    # Shuffle per dataset
    shuffled = {
        dataset: df[df.dataset == dataset]
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
        for dataset in datasets
    }

    slices = []
    for dataset, sdf in shuffled.items():
        slice_df = sdf.iloc[:samples_per_dataset]
        slices.append(slice_df)

    base_df = pd.concat(slices, ignore_index=True)

    # Add decoys drawn from within the dataset
    decoys = base_df.sample(n=NUM_DECOYS, replace=False, random_state=SEED)
    final_df = pd.concat([base_df, decoys], ignore_index=True)

    # Shuffle final dataset
    final_df = final_df.sample(frac=1, random_state=SEED).reset_index(
        drop=True
    )

    # Final formatting
    final_df = final_df[["message_id", "text"]].rename(
        columns={"message_id": "id"}
    )
    final_df.text = final_df.text.apply(make_human_readable)

    # Export
    out_path = output_dir / f"human_annotation_{name}.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Exported {out_path.resolve()}")

    return base_df.index


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

    # Select valid rows
    target_df = df[
        (df.text.str.len() >= 50) & (df.text.str.len() <= 1500)
    ].copy()

    # Build context
    target_df["text"] = get_comments_with_context(
        full_df=df,
        target_df=target_df,
        context_len=2,
    )

    # Shuffle once globally
    target_df = target_df.sample(frac=1, random_state=SEED).reset_index(
        drop=True
    )

    # Split into disjoint halves for A and B
    midpoint = len(target_df) // 2
    df_A = target_df.iloc[:midpoint].copy()
    df_B = target_df.iloc[midpoint:].copy()

    # Build datasets A and B
    build_dataset(df_A, "A", output_dir)
    build_dataset(df_B, "B", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the annotation datasets"
    )
    parser.add_argument(
        "--pefk-path", required=True, help="Path to the full dataset"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for exported datasets"
    )

    args = parser.parse_args()
    main(pefk_path=Path(args.pefk_path), output_dir=Path(args.output_dir))
