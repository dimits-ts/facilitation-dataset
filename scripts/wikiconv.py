from pathlib import Path
import ast

import pandas as pd
from tqdm.auto import tqdm

from tasks import preprocessing_util


INPUT_DIR = Path("../downloads/wikiconv")
INTERMEDIATE_CSV_PATH = Path("../downloads/wikiconv/wikiconv_combined.csv")
OUTPUT_PATH = Path("../datasets/wikiconv.csv")
CHUNK_SIZE = 100_000


def combine_dataset():
    first_chunk = True
    jsonl_files = list(INPUT_DIR.rglob("*.jsonl"))
    for file_path in tqdm(jsonl_files, desc="JSONL files"):
        for chunk in tqdm(
            pd.read_json(file_path, lines=True, chunksize=CHUNK_SIZE),
            total=preprocessing_util.get_num_chunks(file_path, CHUNK_SIZE),
            leave=False,
            desc="JSONL chunks processed",
        ):
            chunk.to_csv(
                INTERMEDIATE_CSV_PATH,
                mode="a",
                index=False,
                header=first_chunk,
            )
            first_chunk = False


def process_dataset():
    print("\t Estimating file size...")
    chunks = preprocessing_util.get_num_chunks(
        INTERMEDIATE_CSV_PATH, CHUNK_SIZE
    )
    print(f"\t Found {chunks} chunks, of {CHUNK_SIZE} rows each.")

    first_chunk = True  # to control writing header only once
    for chunk in tqdm(
        pd.read_csv(INTERMEDIATE_CSV_PATH, chunksize=CHUNK_SIZE),
        total=chunks,
        desc="Chunks processed",
    ):
        df = chunk
        df = df.dropna(subset=["text"])

        # normalize meta json object
        meta_df = pd.json_normalize(df.meta.apply(ast.literal_eval))
        meta_df = meta_df.add_prefix("meta.")

        df = pd.concat(
            [df.reset_index(drop=True), meta_df.reset_index(drop=True)],
            axis=1,
        )
        df = df.drop(columns=["meta"])

        df["is_moderator"] = False
        df["dataset"] = "wikiconv"
        df["notes"] = df.apply(
            lambda row: {
                "toxicity": row.get("meta.toxicity"),
                "severe_toxicity": row.get("meta.sever_toxicity"),
            },
            axis=1,
        )

        df = df.rename(
            columns={
                "conversation_id": "conv_id",
                "reply-to": "reply_to",
                "id": "message_id",
                "speaker": "user",
            }
        )

        df = preprocessing_util.std_format_df(df)

        df.to_csv(
            OUTPUT_PATH, mode="a", index=False, header=first_chunk
        )
        first_chunk = False  # header only for the first chunk


if __name__ == "__main__":
    print("Converting yearly JSONL datasets to unified CSV...")
    combine_dataset()
    print("Processing CSV dataset...")
    process_dataset()
    print("Done.")
