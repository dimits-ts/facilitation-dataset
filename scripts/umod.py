from pathlib import Path

import pandas as pd

from tasks import preprocessing_util


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


def compute_reply_to(
    df, conv_id_col="id", source_col="source", id_col="message_id"
):
    """
    Given a stacked DataFrame of conversation turns with source labels and
    unique IDs, returns a pd.Series `reply_to` where:
        - preceding_comment rows get NaN
        - reply rows get the message_id of their corresponding
        preceding_comment (same conv_id)

    Args:
        df (pd.DataFrame): The stacked DataFrame.
        conv_id_col (str): Column name identifying the conversation.
        source_col (str): Column indicating if a row is a 'preceding_comment'
        or 'reply'.
        id_col (str): Unique identifier for each row (message_id).

    Returns:
        pd.Series: Series of same length as df with reply_to information.
    """
    # Create a mapping from conv_id to message_id for preceding comments
    preceding_map = df.loc[
        df[source_col] == "preceding_comment", [conv_id_col, id_col]
    ]
    conv_to_msgid = dict(
        zip(preceding_map[conv_id_col], preceding_map[id_col])
    )

    # Initialize reply_to as NaNs
    reply_to = pd.Series(index=df.index, dtype=object)

    # Fill in reply rows with the matching preceding_comment message_id
    is_reply = df[source_col] == "reply"
    reply_to[is_reply] = df.loc[is_reply, conv_id_col].map(conv_to_msgid)

    return reply_to


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

    df["message_id"] = df.apply(
            lambda row: preprocessing_util.hash_to_md5(
                row.get("id")
                + row.get("text")
            ),
            axis=1,
        )
    # if comment is reply, is 70% moderation (aggregated via labels) and
    # if annotators are more than 50% confident
    df["is_moderator"] = (
        (df.source == "reply")
        & (df.entropy_moderation <= 0.75)
        & (df.softlabel_raw >= 0.75)
    )
    # all users are unique
    df["user"] = df.message_id
    df["dataset"] = "umod"
    df["reply_to"] = compute_reply_to(df)

    df = df.rename(columns={"id": "conv_id"})
    df = preprocessing_util.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
