from pathlib import Path
import re

import pandas as pd

import util.preprocessing

INPUT_PATH = Path("../../downloads/ceri/ceri.xlsx")
OUTPUT_PATH = Path("../../datasets/ceri.csv")
CLEAN_HTML_PATTERN = re.compile("<.*?>")


def is_moderator(username: str):
    return "moderator" in username.strip().lower()


def rem_html_tags(raw_html: str) -> str:
    return re.sub(CLEAN_HTML_PATTERN, "", raw_html)


def process_sheet(df: pd.DataFrame, sheet_index: int) -> pd.DataFrame:
    df = df.dropna(subset=["COMMENT"]).copy()

    # Make COMMENT ID and COMMENT PARENT unique per sheet
    id_map = {
        orig_id: f"{sheet_index}_{int(orig_id)}"
        for orig_id in df["COMMENT ID"].unique()
    }
    df["COMMENT ID"] = df["COMMENT ID"].map(id_map)
    df["COMMENT PARENT"] = df["COMMENT PARENT"].apply(
        lambda parent_id: id_map.get(parent_id) if parent_id != 0 else None
    )

    df["COMMENT"] = df["COMMENT"].apply(rem_html_tags)
    df["is_moderator"] = df["USER LOGIN"].apply(is_moderator)
    df["moderation_supported"] = True

    # Use POST ID if it exists, otherwise fallback to SUBTOPIC
    if "POST ID" in df.columns and not df["POST ID"].isnull().all():
        conv_source = df["POST ID"].astype(str)
    else:
        conv_source = df["SUBTOPIC"].astype(str)

    df["conv_id"] = conv_source.apply(util.preprocessing.hash_to_md5)
    df["dataset"] = "ceri"
    df["notes"] = None
    df["escalated"] = False
    df["escalation_supported"] = False

    df = df.rename(
        {
            "COMMENT": "text",
            "COMMENT PARENT": "reply_to",
            "USER LOGIN": "user",
            "COMMENT ID": "message_id",
        },
        axis=1,
    )
    return util.preprocessing.std_format_df(df)


def main():
    all_sheets = pd.read_excel(INPUT_PATH, sheet_name=list(range(1, 11)))
    processed_dfs = []

    for idx, df in all_sheets.items():
        processed_df = process_sheet(df, idx)
        processed_dfs.append(processed_df)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
