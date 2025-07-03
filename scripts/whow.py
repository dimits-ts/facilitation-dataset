from pathlib import Path

import pandas as pd

from tasks import preprocessing_util


INPUT_DIR = Path("../downloads/whow/data")
OUTPUT_PATH = Path("../datasets/whow.csv")


def import_df(input_dir: Path) -> pd.DataFrame:
    excel_ls = []
    for excel_file in input_dir.rglob("*.xlsx"):
        df = pd.read_excel(excel_file)
        df["conv_id"] = excel_file.name
        excel_ls.append(df)

    df = pd.concat(excel_ls, ignore_index=True)
    return df


def merge_back_to_back_comments(df: pd.DataFrame) -> pd.DataFrame:
    def extract_group_id(comment_id):
        return comment_id.split("_")[0] if "_" in comment_id else comment_id

    # Extract group ID from the comment ID
    df["message_id"] = df["id"].apply(extract_group_id)

    # Group by the group ID and aggregate selected columns
    merged = df.groupby("message_id", as_index=False).agg(
        {
            "conv_id": "first",
            "speaker": "first",
            "role": "first",
            "text": " ".join,
        }
    )

    return merged


def main():
    df = import_df(INPUT_DIR)
    df = df.astype(str)
    df = merge_back_to_back_comments(df)
    df["is_moderator"] = df.role == "mod"
    df["user"] = df.speaker.apply(preprocessing_util.hash_to_md5)

    # sort by turn order and then run reply_to
    df.message_id = df.message_id.astype(int)
    df = df.sort_values(by="message_id")
    df["reply_to"] = preprocessing_util.assign_reply_to(
        df, conv_id_col="conv_id", message_id_col="message_id"
    )

    df.message_id = "whow-" + df.message_id.astype(str)
    df["dataset"] = "whow"
    df["notes"] = None

    df = preprocessing_util.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
