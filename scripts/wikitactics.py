from pathlib import Path
from typing import Any
import json
import ast

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import preprocessing


INPUT_PATH = Path("../downloads/wikitactics/wikitactics.json")
OUTPUT_PATH = Path("../datasets/wikitactics.csv")


def is_moderator(
    rebuttal_labels: list[str], coordination_labels: list[str]
) -> bool:
    MODERATION_COORDINATION = [
        "Asking questions",
        "Coordinating edits",
        "Providing clarification",
        "Suggesting a compromise",
        "Contextualisation",
    ]
    MODERATION_REBUTTAL = [
        "DH6: Refutation of opponent's argument (with evidence or reasoning)",
        "DH5: Counterargument with new evidence / reasoning",
        "DH7: Refuting the central point",
    ]

    # Check if any label in rebuttal_labels is in MODERATION_REBUTTAL
    if rebuttal_labels is not None and any(
        label in MODERATION_REBUTTAL for label in rebuttal_labels
    ):
        return True

    # Check if any label in coordination_labels is in MODERATION_COORDINATION
    if coordination_labels is not None and any(
        label in MODERATION_COORDINATION for label in coordination_labels
    ):
        return True

    return False


def assign_reply_to(group):
    group = group.sort_index()  # Preserve original order within conversation
    group["reply_to"] = [None] + group["message_id"].iloc[:-1].tolist()
    return group


def main():
    df = pd.read_json(INPUT_PATH)
    df = df.explode(column="utterances")

    utterance_df = pd.json_normalize(df.utterances)
    df = pd.concat([df.reset_index(), utterance_df.reset_index()], axis=1)
    df = df.drop(columns=["utterances", "index"])

    df["is_moderator"] = df.apply(
        lambda row: is_moderator(
            row.get("rebuttal_labels"), row.get("coordination_labels")
        ),
        axis=1,
    )
    df["dataset"] = "wikitactics"
    # make sure message_id is unique across discussions
    df["message_id"] = df.apply(
        lambda row: preprocessing.hash_to_md5(
            row.get("text") + row.get("conv_id")
        ),
        axis=1,
    )
    df["reply_to"] = None  # Initialize the column
    # this is deprecated, but idk how to cleanly keep the conv_id column
    df = df.groupby("conv_id", group_keys=False).apply(assign_reply_to)
    df["notes"] = None
    df = df.rename(columns={"username": "user"})
    df = preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()
