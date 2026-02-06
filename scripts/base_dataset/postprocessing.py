
# The PEFK (Prosocial and Effective Facilitation in Konversations) Dataset
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from ..util import preprocessing


INPUT_DIR = Path("datasets")
OUTPUT_PATH = Path("pefk.csv")


def get_unified_dataset(input_dir: Path) -> pd.DataFrame:
    print(input_dir.resolve())
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
    valid_ids = preprocessing.get_valid_discussion_ids(
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
