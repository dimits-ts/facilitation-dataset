
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


INPUT_PATH = Path(
    "../../downloads/cmv_awry2/conversations-gone-awry-cmv-corpus/"
    "utterances.jsonl"
)
OUTPUT_PATH = Path("../../datasets/cmv_awry2.csv")
PERCENTILE_ESCALATION = 60


def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    deleted_comments = df.text == "[deleted]"
    print(f"Removed {len(deleted_comments)} deleted comments")
    df = df[~deleted_comments]

    # keep pure derailment as note
    df.meta = df.meta.apply(lambda _dict: f"Derailment={_dict['score']}")
    df = df.rename(
        columns={
            "conversation_id": "conv_id",
            "id": "message_id",
            "speaker": "user",
            "reply-to": "reply_to",
            "text": "text",
            "meta": "notes",
        }
    )

    df["dataset"] = "cmv_awry"
    df["escalated"] = df.notes.apply(lambda x: int(x.split("=")[1]))
    threshold = df["escalated"].quantile(PERCENTILE_ESCALATION / 100)
    df["escalated"] = df["escalated"] > threshold
    df["escalation_supported"] = True

    df["is_moderator"] = False
    df["moderation_supported"] = False

    df = preprocessing.std_format_df(df)

    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
