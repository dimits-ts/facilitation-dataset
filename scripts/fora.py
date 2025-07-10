import sys
from pathlib import Path

import pandas as pd

import util.preprocessing


INPUT_PATH = Path("../downloads/fora/corpus_resources/data.csv")
OUTPUT_PATH = Path("../datasets/fora.csv")


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
        df["reply_to"] = util.preprocessing.assign_reply_to(
            df,
            conv_id_col="conversation_id",
            message_id_col="message_id",
            order_col="SpeakerTurn",
        )
        df["notes"] = util.preprocessing.notes_from_columns(
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
        df["escalated"] = False
        df["escalation_supported"] = False
        df["moderation_supported"] = False

        df = df.rename(
            columns={
                "conversation_id": "conv_id",
                "words": "text",
                "speaker_id": "user",
                "is_fac": "is_moderator",
            }
        )
        df = util.preprocessing.std_format_df(df)
        df.to_csv(OUTPUT_PATH, index=False)
    else:
        print("No Fora CSV found, skipping...")
        sys.exit(0)


if __name__ == "__main__":
    main()
