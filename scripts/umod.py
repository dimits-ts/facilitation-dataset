from pathlib import Path

import pandas as pd
import numpy as np

import util.preprocessing


INPUT_PATH = Path("../downloads/umod/umod.csv")
OUTPUT_PATH = Path("../datasets/umod.csv")


def combine_comments(df):
    preceding = df.drop(columns=["reply"]).rename(
        columns={"preceding_comment": "text"}
    )

    # For reply rows
    reply = df.drop(columns=["preceding_comment"]).rename(
        columns={"reply": "text"}
    )

    # Add a helper column to keep track of source (optional)
    preceding["source"] = "preceding_comment"
    reply["source"] = "reply"

    # Concatenate vertically
    combined = pd.concat([preceding, reply], ignore_index=True)
    return combined


def aggregate_notes(df, exclude_cols):
    """
    Given a stacked DataFrame with columns including conv_id, text, source,
    aggregate all other columns into a single 'notes' dictionary column.

    Args:
        df (pd.DataFrame): Input stacked DataFrame.
        conv_id_col (str): Name of conversation ID column.
        text_col (str): Name of the text column.
        source_col (str): Name of the source column.

    Returns:
        pd.DataFrame: DataFrame with columns [conv_id, text, source, notes]
    """
    notes_cols = [col for col in df.columns if col not in exclude_cols]

    df = df.copy()
    df["notes"] = df[notes_cols].apply(
        lambda row: row.dropna().to_dict(), axis=1
    )
    df = df.drop(columns=notes_cols)

    return df


def main():
    df = pd.read_csv(INPUT_PATH, sep="\t")
    df = combine_comments(df)
    df = aggregate_notes(
        df,
        exclude_cols=[
            "id",
            "entropy_moderation",
            "text",
            "source",
            "softlabel_raw",
        ],
    )
    df["speaker_turn"] = np.where(df.source == "reply", 1, 0)
    df["message_id"] = df.apply(
        lambda row: f"umod-{row.get('id')}-{row.get('speaker_turn')}",
        axis=1,
    )
    # if comment is reply, is 70% moderation (aggregated via labels) and
    # if annotators are more than 50% confident
    df["is_moderator"] = (
        (df.source == "reply")
        & (df.entropy_moderation <= 0.75)
        & (df.softlabel_raw >= 0.75)
    )
    df["moderation_supported"] = True

    # all users are unique
    df["user"] = "user-" + df.message_id

    df["dataset"] = "umod"
    df["reply_to"] = util.preprocessing.assign_reply_to(
        df,
        conv_id_col="id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )
    df["escalated"] = False
    df["escalation_supported"] = False

    df = df.rename(columns={"id": "conv_id"})
    df = util.preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
