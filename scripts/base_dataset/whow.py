
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


INPUT_DIR = Path("../../downloads/whow/data")
OUTPUT_PATH = Path("../../datasets/whow.csv")
DIALOGUE_ACTS = [
    "probing",
    "confronting",
    "instruction",
    "interpretation",
    "supplement",
    "utility",
]


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
            "dialogue act": ",".join,
        }
    )
    merged = merged.rename(
        columns={
            "dialogue act": "dialogue_act",
        }
    )

    return merged


def expand_dialogue_acts(df: pd.DataFrame) -> pd.DataFrame:
    # for each row where is_moderator is True
    # for each value in the dialogue_act string
    # select unique values
    # match each value (which is an index on the DIALOGUE_PATH array)
    # then create columns for each of the values and set it to 1 where
    #  the index is present

    # Ensure dialogue_act is a string and handle missing values
    df.dialogue_act = df.dialogue_act.fillna("").astype(str)

    # Initialize new columns for all dialogue acts with False
    for act in DIALOGUE_ACTS:
        df[act] = False

    # Only expand for moderator rows
    moderator_rows = df["is_moderator"]

    # Go through each row and set True where the dialogue act index appears
    for idx, row in df.loc[moderator_rows].iterrows():
        acts_str = row["dialogue_act"]
        if acts_str.strip() == "":
            continue

        # Split by comma, strip spaces, and filter out empties
        act_indices = [
            a.strip() for a in acts_str.split(",") if a.strip().isdigit()
        ]

        # Set the corresponding columns to True if index exists
        for a in act_indices:
            act_idx = int(a)
            if 0 <= act_idx < len(DIALOGUE_ACTS):
                df.at[idx, DIALOGUE_ACTS[act_idx]] = True

    return df


def main():
    df = import_df(INPUT_DIR)
    df = df.astype(str)

    df = merge_back_to_back_comments(df)

    df["is_moderator"] = df.role == "mod"
    df["moderation_supported"] = True

    df["user"] = df.speaker.apply(preprocessing.hash_to_md5)

    df["speaker_turn"] = df.groupby("conv_id").cumcount() + 1
    df["message_id"] = df.apply(
        lambda row: f"whow-{row.get('conv_id')}-{row.get('speaker_turn')}",
        axis=1,
    )
    df = df.sort_values(by="message_id")
    df["reply_to"] = preprocessing.assign_reply_to(
        df,
        conv_id_col="conv_id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )

    df = expand_dialogue_acts(df)
    df["notes"] = preprocessing.notes_from_columns(
        df,
        DIALOGUE_ACTS,
    )

    df["dataset"] = "whow"
    df["escalated"] = False
    df["escalation_supported"] = False

    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
