from pathlib import Path
import typing
import inspect

import pandas as pd
import nltk
import DiscQuA
from tqdm.auto import tqdm

import util.io


DATASET_PATH = Path("pefk.csv")
OUTPUT_DIR = Path("discqua")
METRICS = [
    DiscQuA.calculate_structure_features,
    DiscQuA.calculate_balanced_participation,
    DiscQuA.calculate_language_features,
    DiscQuA.calculate_coordination_per_disc_utt,
    DiscQuA.calculate_politeness,
    DiscQuA.calculate_readability,
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

    all_args = {
        "message_list": get_message_list(conv),
        "message_id_list": get_message_id_list(conv),
        "speakers_list": get_speakers_list(conv),
        "replyto_list": get_reply_to_list(conv),
        "disc_id": conv_id,
        "discussion_level": False,
        "ctx": 3
    }

    # Provide only accepted parameters, since only a subset is used each time
    sig = inspect.signature(metric)
    accepted_args = {k: v for k, v in all_args.items() if k in sig.parameters}
    return metric(**accepted_args)


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
