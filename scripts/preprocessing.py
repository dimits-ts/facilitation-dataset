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
    - min_comments (int): Minimum number of comments required to keep a discussion.
    - max_comments (int or None): Maximum number of comments allowed to keep a discussion.
    - discussion_col (str): Name of the column identifying discussions.

    Returns:
    - pd.DataFrame: Filtered dataframe containing only discussions within the specified range.
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
