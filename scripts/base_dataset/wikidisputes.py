
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
from typing import Any
import json

import pandas as pd

from ..util import preprocessing


INPUT_DIR = Path("downloads/wikidisputes/data")
OUTPUT_PATH = Path("datasets/wikidisputes.csv")


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
    df = df[df["conversation.text"].apply(len) > 2]

    # remove duplicate comments
    df_duplicate = df[df.duplicated("conversation.id", keep=False)]
    df_duplicate_convs = set(df_duplicate["conversation.id"].unique())
    df = df[~df["conversation.id"].isin(df_duplicate_convs)]
    print(f"Removed {len(df_duplicate)} comments with duplicate ids.")

    df["is_moderator"] = df["conversation.user"] == df["dispute.mediator"]
    df["moderation_supported"] = False

    df["dataset"] = "wikidisputes"

    df["conversation.id"] = "wikidisputes-" + df["conversation.id"]
    df["conversation.reply_to"] = "wikidisputes-" + df["conversation.reply_to"]
    # duplicate values in message_id
    df = df.rename(
        columns={
            "conversation.conv_id": "conv_id",
            "conversation.user": "user",
            "conversation.reply_to": "reply_to",
            "conversation.text": "text",
            "conversation.id": "message_id",
        }
    )
    df["notes"] = preprocessing.notes_from_columns(
        df,
        ["toxicity", "severe_toxicity"],
    )
    # df.escalated already in dataframe
    df["escalation_supported"] = True
    df = preprocessing.std_format_df(df)

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
