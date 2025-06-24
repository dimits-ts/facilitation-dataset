from pathlib import Path
import json

import pandas as pd

from tasks import preprocessing_util


INPUT_PATH = Path("../downloads/iq2/iq2_data_release.json")
OUTPUT_PATH = Path("../datasets/iq2.csv")


def json_to_df(json_dataset: dict) -> pd.DataFrame:
    discussion_dfs = []
    for discussion_name in json_dataset.keys():
        df = pd.json_normalize(json_dataset[discussion_name]["transcript"])
        df = df.loc[:, ["speakertype", "speaker", "paragraphs"]]
        df["conv_id"] = discussion_name
        discussion_dfs.append(df)
    return pd.concat(discussion_dfs, ignore_index=True)


def assign_reply_to(group):
    group = group.sort_index()  # Preserve original order within conversation
    group["reply_to"] = [None] + group["message_id"].iloc[:-1].tolist()
    return group


def main():
    with INPUT_PATH.open("r") as fin:
        contents = json.load(fin)

    df = json_to_df(contents)
    df = df.explode("paragraphs")
    df.paragraphs = df.paragraphs.str.replace("uh,", "")
    df.paragraphs = df.paragraphs.str.replace("...", "")

    df["is_moderator"] = df.speakertype.apply(lambda x: x in ["mod", "host"])
    df["message_id"] = df.apply(
        lambda row: preprocessing_util.hash_to_md5(
            row.get("paragraphs") + row.get("conv_id")
        ),
        axis=1,
    )
    df["dataset"] = "iq2"
    df["notes"] = None
    df["reply_to"] = None  # Initialize the column
    # this is deprecated, but idk how to cleanly keep the conv_id column
    df = df.groupby("conv_id", group_keys=False).apply(assign_reply_to)

    df = df.rename(
        columns={
            "paragraphs": "text",
            "speaker": "user",
        }
    )
    df = preprocessing_util.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
