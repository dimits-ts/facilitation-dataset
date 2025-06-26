from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from tasks import preprocessing_util


INPUT_DIR = Path("../datasets")
OUTPUT_PATH = Path("../pefk.csv")


def get_unified_dataset(input_dir: Path) -> pd.DataFrame:
    for file_path in tqdm(input_dir.rglob("*.csv"), desc="Combining datasets"):
        dfs = []
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    full_df = pd.concat(dfs)
    return full_df


def discard_one_man_convs(df: pd.DataFrame) -> pd.DataFrame:
    valid_ids = preprocessing_util.get_valid_discussion_ids(
        df, conv_id_col="conv_id", user_col="user"
    )
    print(f"Keeping {len(valid_ids)} valid comments, out of {df.shape[0]}.")
    df = df[df.conv_id.isin(valid_ids)]
    return df


def main():
    df = get_unified_dataset(INPUT_DIR)
    df = discard_one_man_convs(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset exported as {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
