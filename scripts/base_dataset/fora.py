
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

import sys
from pathlib import Path

import pandas as pd

from ..util import preprocessing


INPUT_PATH = Path("downloads/fora/corpus_resources/data.csv")
OUTPUT_PATH = Path("datasets/fora.csv")


def main():
    if INPUT_PATH.exists():
        df = pd.read_csv(INPUT_PATH)
        df.words = df.words.astype(str)
        df.conversation_id = df.conversation_id.astype(str)
        df.SpeakerTurn = df.SpeakerTurn.astype(int)
        df["message_id"] = df.apply(
            lambda row: f"fora-{row.get('conversation_id')}"
            f"-{row.get('SpeakerTurn')}",
            axis=1,
        )
        df["reply_to"] = preprocessing.assign_reply_to(
            df,
            conv_id_col="conversation_id",
            message_id_col="message_id",
            order_col="SpeakerTurn",
        )
        df["notes"] = preprocessing.notes_from_columns(
            df,
            [
                "Express affirmation",
                "Specific invitation",
                "Provide example",
                "Open invitation",
                "Make connections",
                "Express appreciation",
                "Follow up question",
            ],
        )
        df["dataset"] = "fora"
        df["escalated"] = False
        df["escalation_supported"] = False
        df["moderation_supported"] = True

        df = df.rename(
            columns={
                "conversation_id": "conv_id",
                "words": "text",
                "speaker_id": "user",
                "is_fac": "is_moderator",
            }
        )
        df = preprocessing.std_format_df(df)
        df.to_csv(OUTPUT_PATH, index=False)
    else:
        print("No Fora CSV found, skipping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
