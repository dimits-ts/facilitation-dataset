
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

from ..util import preprocessing


INPUT_PATH = Path("downloads/wikitactics/wikitactics.json")
OUTPUT_PATH = Path("datasets/wikitactics.csv")


def is_moderator(
    rebuttal_labels: list[str], coordination_labels: list[str]
) -> bool:
    MODERATION_COORDINATION = [
        "Asking questions",
        "Coordinating edits",
        "Providing clarification",
        "Suggesting a compromise",
        "Contextualisation",
    ]

    # Check if any label in coordination_labels is in MODERATION_COORDINATION
    if coordination_labels is not None and any(
        label in MODERATION_COORDINATION for label in coordination_labels
    ):
        return True

    return False


def main():
    df = pd.read_json(INPUT_PATH)
    df = df.explode(column="utterances")

    utterance_df = pd.json_normalize(df.utterances)
    df = pd.concat([df.reset_index(), utterance_df.reset_index()], axis=1)
    df = df.drop(columns=["utterances", "index"])

    df["is_moderator"] = df.apply(
        lambda row: is_moderator(
            row.get("rebuttal_labels"), row.get("coordination_labels")
        ),
        axis=1,
    )
    df["moderation_supported"] = True

    df["speaker_turn"] = df.groupby("conv_id").cumcount() + 1
    # make sure message_id is unique across discussions
    df["message_id"] = df.apply(
        lambda row: f"wikitactics-{row.get('conv_id')}-"
        f"{row.get('speaker_turn')}",
        axis=1,
    )
    df["reply_to"] = preprocessing.assign_reply_to(
        df,
        conv_id_col="conv_id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )
    df["dataset"] = "wikitactics"
    df["notes"] = None
    df["escalated"] = df["escalation_label"]
    df["escalation_supported"] = True
    df = df.rename(columns={"username": "user"})
    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
