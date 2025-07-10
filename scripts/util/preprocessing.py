import hashlib
import subprocess
import math
from pathlib import Path

import pandas as pd


def filter_discussions_by_comment_count(
    df: pd.DataFrame,
    discussion_col: str,
    min_comments: int = 1,
    max_comments: int = 1000,
) -> pd.DataFrame:
    """
    Filters discussions based on the number of comments, keeping only those
    with a number of comments between min_comments and max_comments.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the discussion data.
    - min_comments (int): Minimum number of comments required to keep a
    discussion.
    - max_comments (int or None): Maximum number of comments allowed to keep a
    discussion.
    - discussion_col (str): Name of the column identifying discussions.

    Returns:
    - pd.DataFrame: Filtered dataframe containing only discussions within the
    specified range.
    """
    discussion_counts = df[discussion_col].value_counts()

    if max_comments is not None:
        valid_discussions = discussion_counts[
            (discussion_counts >= min_comments)
            & (discussion_counts <= max_comments)
        ].index
    else:
        valid_discussions = discussion_counts[
            discussion_counts >= min_comments
        ].index

    filtered_df = df[df[discussion_col].isin(valid_discussions)].copy()
    return filtered_df


def hash_to_md5(input_string: str) -> str:
    """
    Hashes a string using MD5 and returns the hexadecimal digest.

    Args:
        input_string (str): The input string to hash.

    Returns:
        str: The MD5 hash of the input string in hexadecimal format.
    """
    md5_hash = hashlib.md5(input_string.encode("utf-8"))
    return md5_hash.hexdigest()


def std_format_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[
        :,
        [
            "conv_id",
            "message_id",
            "reply_to",
            "user",
            "is_moderator",
            "moderation_supported",
            "escalated",
            "escalation_supported",
            "text",
            "dataset",
            "notes",
        ],
    ]


def get_num_chunks(file_path: Path, chunk_size: int) -> int:
    result = subprocess.run(
        ["wc", "-l", str(file_path)], capture_output=True, text=True
    )
    return math.ceil(int(result.stdout.strip().split()[0]) / chunk_size)


def get_valid_discussion_ids(df, conv_id_col: str, user_col: str):
    user_counts = df.groupby(conv_id_col)[user_col].nunique()
    valid_discussions = user_counts[user_counts > 1]
    return valid_discussions.index.tolist()


def assign_reply_to(
    df: pd.DataFrame, conv_id_col: str, message_id_col: str, order_col: str
) -> pd.Series:
    df_sorted = df.sort_values([conv_id_col, order_col])
    # shift comment id by 1
    reply_to = df_sorted.groupby(conv_id_col)[message_id_col].shift(1)
    # The result is aligned with df_sorted, we must reindex to original
    # df order
    reply_to = reply_to.reindex(df.index)
    return reply_to


def notes_from_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df.apply(
        lambda row: {col: row.get(col) for col in cols},
        axis=1,
    )
