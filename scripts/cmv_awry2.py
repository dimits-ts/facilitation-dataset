from pathlib import Path

import pandas as pd

import util.preprocessing


INPUT_PATH = Path(
    "../downloads/cmv_awry2/conversations-gone-awry-cmv-corpus/"
    "utterances.jsonl"
)
OUTPUT_PATH = Path("../datasets/cmv_awry2.csv")


def main():
    df = pd.read_json(INPUT_PATH, lines=True)

    deleted_comments = df.text == "[deleted]"
    print(f"Removed {len(deleted_comments)} deleted comments")
    df = df[~deleted_comments]

    df["dataset"] = "cmv_awry"
    df["is_moderator"] = False
    
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
    df = util.preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
