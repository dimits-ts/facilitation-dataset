from pathlib import Path
import json

import pandas as pd

from ..util import preprocessing


INPUT_PATH = Path("../../downloads/iq2/iq2_data_release.json")
OUTPUT_PATH = Path("../../datasets/iq2.csv")


def json_to_df(json_dataset: dict) -> pd.DataFrame:
    discussion_dfs = []
    for discussion_name in json_dataset.keys():
        df = pd.json_normalize(json_dataset[discussion_name]["transcript"])
        df = df.loc[:, ["speakertype", "speaker", "paragraphs"]]
        df["conv_id"] = discussion_name
        discussion_dfs.append(df)
    return pd.concat(discussion_dfs, ignore_index=True)


def main():
    with INPUT_PATH.open("r") as fin:
        contents = json.load(fin)

    df = json_to_df(contents)
    df = df.explode("paragraphs")
    df = df.reset_index()

    df["is_moderator"] = df.speakertype.apply(lambda x: x in ["mod", "host"])
    df["moderation_supported"] = True

    df["speaker_turn"] = df.groupby("conv_id").cumcount() + 1
    df["message_id"] = df.apply(
        lambda row: f"iq2-{row.get('conv_id')}-{row.get('speaker_turn')}",
        axis=1,
    )
    df["dataset"] = "iq2"
    df["notes"] = None

    df["escalated"] = False
    df["escalation_supported"] = False
    
    df["reply_to"] = preprocessing.assign_reply_to(
        df,
        conv_id_col="conv_id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )

    df = df.rename(
        columns={
            "paragraphs": "text",
            "speaker": "user",
        }
    )
    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
