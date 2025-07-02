import sys
from pathlib import Path

import pandas as pd
import numpy as np

from tasks import preprocessing_util


INPUT_PATH = Path("../downloads/fora/corpus_resources/data.csv")
OUTPUT_PATH = Path("../datasets/fora.csv")


def main():
    if INPUT_PATH.exists():
        df = pd.read_csv(INPUT_PATH)
        df.words = df.words.astype(str)
        df.collection_id = df.collection_id.astype(str)
        df["message_id"] = df.apply(
            lambda row: preprocessing_util.hash_to_md5(
                row.get("collection_id") + row.get("words")
            ),
            axis=1,
        )
        df["reply_to"] = preprocessing_util.assign_reply_to(df, conv_id_col="collection_id", message_id_col="message_id")
        df["notes"] = preprocessing_util.notes_from_columns(df, ["Personal story", "Personal experience", "Express affirmation", "Specific invitation", "Provide example", "Open invitation", "Make connections", "Express appreciation", "Follow up question"])
        df["dataset"] = "fora"

        df = df.rename(
            columns={
                "collection_id": "conv_id", 
                "words": "text", 
                "speaker_id": "user",
                "is_fac": "is_moderator"
                }
            )
        df = preprocessing_util.std_format_df(df)
        df.to_csv(OUTPUT_PATH, index=False)
    else:
        print("No Fora CSV found, skipping...")
        sys.exit(0)


if __name__ == "__main__":
    main()