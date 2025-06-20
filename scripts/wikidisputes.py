from pathlib import Path
from typing import Any
import json

import pandas as pd

import preprocessing


INPUT_DIR = Path("../downloads/wikidisputes/data")
OUTPUT_PATH = Path("../datasets/wikidisputes.csv")


def read_json(path: Path) -> pd.DataFrame:
    with path.open("r") as fin:
        contents: list[dict[str, Any]] = json.load(fin)

    df = pd.concat([pd.json_normalize(record) for record in contents])
    return df


def main():
    escalated_df = read_json(INPUT_DIR / "escalated.json")
    escalated_df["escalated"] = True
    not_escalated_df = read_json(INPUT_DIR / "not_escalated.json")
    not_escalated_df["escalated"] = False
    df = pd.concat([escalated_df, not_escalated_df])
    # explode list of conversations
    df = df.explode(column="conversation")

    # normalize conversation json object
    conv_df = pd.json_normalize(df.conversation)
    conv_df = conv_df.add_prefix("conversation.")

    df = pd.concat([df.reset_index(), conv_df.reset_index()], axis=1)
    df = df.drop(columns=["conversation", "index"])
    df = df[df["conversation.type"] == "original"]

    df["is_moderator"] = df["conversation.user"] == df["dispute.mediator"]
    df["dataset"] = "wikidisputes"
    df["notes"] = df.apply(
        lambda row: {
            "escalated": row["escalated"],
            "toxicity": row["conversation.toxicity"],
            "severe_toxicity": row["conversation.severe_toxicity"],
        },
        axis=1,
    )

    df = df.rename(
        columns={
            "conversation.conv_id": "conv_id",
            "conversation.user": "user",
            "conversation.reply_to": "reply_to",
            "conversation.text": "text",
            "conversation.id": "message_id",
        }
    )
    df = preprocessing.std_format_df(df)

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
