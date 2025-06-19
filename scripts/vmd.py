from pathlib import Path
from typing import Any
import json
import ast

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import preprocessing

INPUT_PATH = Path("../downloads/vmd/data/datasets/dataset.csv")
OUTPUT_PATH = Path("../datasets/vmd.csv")


def get_toxicity(annotation_str: str) -> float:
    try:
        return float(annotation_str.split("\n")[0][-1])
    except ValueError:
        return float("nan")


def get_arg_qual(annotation_str: str) -> float:
    try:
        return float(annotation_str.split("\n")[1][-1])
    except ValueError:
        return float("nan")


def assign_reply_to(df):
    # Sort by conv_id and some timestamp proxy, here assuming row order or message_id order
    df = df.sort_values(by=["conv_id", "message_id"]).reset_index(drop=True)

    # Create a new column with shifted message_ids within each conversation
    df["reply_to"] = df.groupby("conv_id")["message_id"].shift(1)

    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    df.message_id = df.message_id.astype(str)

    df["toxicity"] = df.annotation.apply(get_toxicity)
    df["arg_qual"] = df.annotation.apply(get_arg_qual)
    df = df.groupby(
        ["conv_id", "message_id", "user", "message", "is_moderator"]
    ).agg({"toxicity": "mean", "arg_qual": "mean"})
    df = df.reset_index()

    df = assign_reply_to(df)
    df["reply_to"] = df.reply_to.astype(str)

    df["notes"] = df.apply(
        lambda row: {
            "model": row.get("model"),
            "toxicity": row.get("toxicity"),
            "arg_qual": row.get("arg_qual"),
        },
        axis=1,
    )
    df["dataset"] = "vmd"
    df = df.rename(columns={"message": "text"})
    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()