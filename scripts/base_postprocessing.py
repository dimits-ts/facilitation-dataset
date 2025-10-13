from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

import util.preprocessing


INPUT_DIR = Path("../datasets")
OUTPUT_PATH = Path("../pefk.csv")
MAX_LENGTH_WORDS = 1000


def get_unified_dataset(input_dir: Path) -> pd.DataFrame:
    dfs = []
    for file_path in tqdm(
        list(input_dir.rglob("*.csv")), desc="Combining datasets"
    ):
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def discard_one_man_convs(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing non-discussion entries...")
    valid_ids = util.preprocessing.get_valid_discussion_ids(
        df, conv_id_col="conv_id", user_col="user"
    )
    print(f"Keeping {len(valid_ids)} valid discussions, out of {df.shape[0]}.")
    df = df[df.conv_id.isin(valid_ids)]
    return df


def find_duplicate_comments(
    df: pd.DataFrame, original_dataset: str, duplicate_dataset: str
) -> pd.Series:
    print(
        "Removing inter-dataset duplicate comments "
        f"(original dataset: '{original_dataset}' -> "
        f"duplicate dataset: '{duplicate_dataset}')..."
    )
    original_dataset_df = df[df.dataset == original_dataset]
    duplicate_dataset_df = df[df.dataset == duplicate_dataset]

    original_keys = set(original_dataset_df.conv_id)
    duplicate_mask = duplicate_dataset_df.conv_id.apply(
        lambda x: x in original_keys
    )
    return duplicate_dataset_df.loc[duplicate_mask, "message_id"].tolist()


def discard_duplicates(
    df: pd.DataFrame, original_dataset: str, duplicate_dataset: str
) -> pd.DataFrame:
    initial_size = len(df)

    keys = set(
        find_duplicate_comments(
            df,
            original_dataset=original_dataset,
            duplicate_dataset=duplicate_dataset,
        )
    )
    df = df[~df.message_id.isin(keys)]

    print(f"Removed {initial_size - len(df)} duplicate comments.")
    return df


def discard_empty_comments(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing empty comments...")
    initial_size = len(df)
    df = df[df.text.progress_apply(lambda x: x.strip()).apply(len) != 0]
    print(f"Removed {initial_size - len(df)} empty comments.")
    return df


def truncate_long_comments(
    df: pd.DataFrame, max_length_words: int
) -> pd.DataFrame:
    print(f"Truncating extremely long comments (>{max_length_words} words)...")
    initial_size = len(df)
    df["text"] = df["text"].progress_apply(
        lambda x: _truncate_text(x, max_length_words)
    )

    num_truncated = sum(df["text"].str.endswith(" [...]"))
    print(f"Truncated {num_truncated} long comments (out of {initial_size}).")
    return df


def _truncate_text(text: str, max_length_words: int) -> str:
    words = text.split()
    if len(words) > max_length_words:
        return " ".join(words[:max_length_words]) + " [...]"
    return text


def discard_nan_comments(df: pd.DataFrame) -> pd.DataFrame:
    print("Removing NaN comments...")
    initial_size = len(df)
    df = df.dropna(subset=["text", "message_id", "conv_id"])
    print(f"Removed {initial_size - len(df)} NaN comments.")
    return df


def main():
    tqdm.pandas()
    df = get_unified_dataset(INPUT_DIR)
    df.text = df.text.astype(str)

    if (
        "wikiconv" in df.dataset.unique()
        and "wikitactics" in df.dataset.unique()
    ):
        df = discard_duplicates(
            df,
            original_dataset="wikitactics",
            duplicate_dataset="wikiconv",
        )

    if (
        "wikiconv" in df.dataset.unique()
        and "wikidisputes" in df.dataset.unique()
    ):
        df = discard_duplicates(
            df,
            original_dataset="wikidisputes",
            duplicate_dataset="wikiconv",
        )

    df = truncate_long_comments(df, max_length_words=MAX_LENGTH_WORDS)
    df = discard_empty_comments(df)
    df = discard_nan_comments(df)
    df = discard_one_man_convs(df)
    # Shift one row *up* within each conversation
    df["should_intervene"] = df.groupby("conv_id")["is_moderator"].shift(-1)
    print(
        "Post-processing complete. "
        f"Exporting dataset to {OUTPUT_PATH.resolve()}..."
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset exported as {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
