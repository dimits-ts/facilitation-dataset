import sys
from pathlib import Path

import pandas as pd

from tasks import preprocessing_util


INPUT_PATH = Path("../downloads/fora/corpus_resources/data.csv")
OUTPUT_PATH = Path("../datasets/fora.csv")


def main():
    if INPUT_PATH.exists():
        df = pd.read_csv(INPUT_PATH)
        df.words = df.words.astype(str)
        df.conversation_id = df.conversation_id.astype(str)
        df.SpeakerTurn = df.SpeakerTurn.astype(str)
        df["message_id"] = df.apply(
            lambda row: preprocessing_util.hash_to_md5(
                row.get("conversation_id")
                + row.get("words")
                + row.get("SpeakerTurn")
            ),
            axis=1,
        )
        df["reply_to"] = preprocessing_util.assign_reply_to(
            df, conv_id_col="conversation_id", message_id_col="message_id"
        )
        df["notes"] = preprocessing_util.notes_from_columns(
            df,
            [
                "Personal story",
                "Personal experience",
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

        df = df.rename(
            columns={
                "conversation_id": "conv_id",
                "words": "text",
                "speaker_id": "user",
                "is_fac": "is_moderator",
            }
        )
        df = preprocessing_util.std_format_df(df)
        df.to_csv(OUTPUT_PATH, index=False)
    else:
        print("No Fora CSV found, skipping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
