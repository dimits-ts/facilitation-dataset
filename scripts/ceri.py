from pathlib import Path
import re

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import preprocessing

INPUT_PATH = Path("../downloads/ceri/ceri.xlsx")
OUTPUT_PATH = Path("../datasets/ceri.csv")
CLEAN_HTML_PATTERN = re.compile("<.*?>")


def is_moderator(username: str):
    return "moderator" in username.strip().lower()


# https://stackoverflow.com/a/12982689
def rem_html_tags(raw_html: str) -> str:
    cleantext = re.sub(CLEAN_HTML_PATTERN, "", raw_html)
    return cleantext

def main():
    df_dict = pd.read_excel(
        INPUT_PATH,
        sheet_name=list(range(1, 11)),
    )
    df = pd.concat(df_dict)
    df = df.dropna(subset="COMMENT")

    df["COMMENT"] = df["COMMENT"].apply(rem_html_tags)
    df["is_moderator"] = df["USER LOGIN"].apply(is_moderator)
    df["POST ID"] = df["POST ID"].astype(str).apply(preprocessing.hash_to_md5)
    df["COMMENT PARENT"] = df["COMMENT PARENT"].apply(
        lambda _id: None if _id == 0 else _id
    )
    df["dataset"] = "CeRI"
    df["notes"] = None

    df = df.rename(
        {
            "POST ID": "conv_id",
            "COMMENT": "text",
            "COMMENT PARENT": "reply_to",
            "COMMENT ID": "message_id",
            "USER ID": "user",
        },
        axis=1,
    )
    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()