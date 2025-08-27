from pathlib import Path
import typing

import pandas as pd
import nltk
import DiscQua
from tqdm.auto import tqdm

import util.io


DATASET_PATH = Path("pefk.csv")
OUTPUT_DIR = Path("discqua")
METRICS = [
    DiscQua.structure_features,
    DiscQua.balanced_participation,
    DiscQua.language_features,
    DiscQua.coordination_per_disc_utt,
    DiscQua.politeness,
    DiscQua.readability,
    DiscQua.collaboration,
]


def get_message_list(discussion: pd.DataFrame) -> list[str]:
    return discussion.text.to_list()


def get_speakers_list(discussion: pd.DataFrame) -> list[str]:
    return discussion.user.to_list()


def get_message_id_list(discussion: pd.DataFrame) -> list[str]:
    return discussion.message_id.to_list()


def get_reply_to_list(discussion: pd.DataFrame) -> list[str]:
    return discussion.reply_to.to_list()


def calculate_metric(
    df: pd.DataFrame, conv_id: str, metric: typing.Callable
) -> float:
    conv = df[df.conv_id == conv_id]

    return metric(
        message_list=get_message_list(conv),
        message_id_list=get_message_id_list(conv),
        speakers_list=get_speakers_list(conv),
        replyto_list=get_reply_to_list(conv),
        disc_id=conv_id,
        discussion_level=False,
    )


def main():
    nltk.download("punk_tab")
    nltk.download("stopwords")

    df = util.io.progress_load_csv(DATASET_PATH)
    results = []
    for metric in tqdm(METRICS):
        for conv_id in tqdm(df.conv_id.unique()):
            result = calculate_metric(df, conv_id, metric)
            results.append((metric, conv_id, result))
            print(result)


if __name__ == "__main__":
    main()
