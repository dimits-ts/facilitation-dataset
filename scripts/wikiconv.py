from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from tasks import preprocessing_util


INPUT_DIR = Path("../downloads/wikiconv")
OUTPUT_PATH = Path("../datasets/wikiconv.csv")
CHUNK_SIZE = 100_000


def add_notes(df):
    # normalize meta json object
    meta_df = pd.json_normalize(df.meta)
    meta_df = meta_df.add_prefix("meta.")

    df = pd.concat(
        [df.reset_index(drop=True), meta_df.reset_index(drop=True)],
        axis=1,
    )
    df = df.drop(columns=["meta"])
    df["notes"] = df.apply(
        lambda row: {
            "toxicity": row.get("meta.toxicity"),
            "severe_toxicity": row.get("meta.sever_toxicity"),
        },
        axis=1,
    )
    return df


def add_meta_cols(df):
    df["is_moderator"] = False
    df["dataset"] = "wikiconv"
    return df


def conform_to_pefk(df):
    df = df.rename(
        columns={
            "conversation_id": "conv_id",
            "reply-to": "reply_to",
            "id": "message_id",
            "speaker": "user",
        }
    )

    df = preprocessing_util.std_format_df(df)
    return df


def process_dataset(df):
    df = df.dropna(subset=["text"])
    df = add_notes(df)
    df = add_meta_cols(df)
    df = conform_to_pefk(df)
    return df


def main():
    first_chunk = True
    jsonl_files = list(INPUT_DIR.rglob("*.jsonl"))
    for file_path in tqdm(jsonl_files, desc="Yearly datasets"):
        for chunk in tqdm(
            pd.read_json(file_path, lines=True, chunksize=CHUNK_SIZE),
            total=preprocessing_util.get_num_chunks(file_path, CHUNK_SIZE),
            leave=False,
            desc="Dataset chunks",
        ):
            df = process_dataset(chunk)
            df.to_csv(
                OUTPUT_PATH,
                mode="a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False


if __name__ == "__main__":
    main()
