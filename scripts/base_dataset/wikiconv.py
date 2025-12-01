from pathlib import Path
import uuid

import pandas as pd
import py3langid as langid
from tqdm.auto import tqdm

import util.preprocessing
import util.io


INPUT_DIR = Path("../../downloads/wikiconv")
OUTPUT_PATH = Path("../../datasets/wikiconv.csv")
CHUNK_SIZE = 100_000


def main():
    first_chunk = True
    jsonl_files = list(INPUT_DIR.rglob("*.jsonl"))
    for file_path in tqdm(jsonl_files, desc="Yearly datasets"):
        for chunk in tqdm(
            pd.read_json(file_path, lines=True, chunksize=CHUNK_SIZE),
            total=util.io.get_num_chunks(file_path, CHUNK_SIZE),
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


def process_dataset(full_df):
    df = full_df.loc[
        full_df.text.apply(lambda x: x.strip()).apply(len) > 0
    ].copy()
    # masked IP addresses are tracked to the same user_id
    # (found in wikiconv-20**/speakers.json). Thus, to be safe, we consider
    # them as separate, unique users
    df.speaker = df.speaker.astype(str).apply(
        lambda x: (
            x if not x.endswith("xxx") else f"unknown-user-{uuid.uuid4().int}"
        )
    )

    tqdm.pandas(desc="Detecting language", leave=False)
    df = df[df.text.astype(str).progress_apply(is_english)]

    # Filter out conversations with only one user commenting
    valid_discussion_ids = util.preprocessing.get_valid_discussion_ids(
        df, conv_id_col="conversation_id", user_col="speaker"
    )
    df = df[df.conversation_id.isin(valid_discussion_ids)]

    df = add_notes(df)
    df = add_meta_cols(df)
    df = conform_to_pefk(df)
    return df


def add_notes(df):
    # normalize meta json object
    meta_df = pd.json_normalize(df.meta)
    meta_df = meta_df.add_prefix("meta.")

    df = pd.concat(
        [df.reset_index(drop=True), meta_df.reset_index(drop=True)],
        axis=1,
    )
    df = df.drop(columns=["meta"])
    df["notes"] = util.preprocessing.notes_from_columns(
        df, ["meta.toxicity", "meta.sever_toxicity"]
    )
    return df


def add_meta_cols(df):
    df["is_moderator"] = False
    df["moderation_supported"] = False
    # TODO: Maybe infer esclation from toxicity using statistical relationships
    # between toxicity and escalation from wikitactics
    df["escalated"] = False
    df["escalation_supported"] = False

    df["dataset"] = "wikiconv"
    return df


def conform_to_pefk(df):
    df = df.rename(
        columns={
            "conversation_id": "conv_id",
            "reply-to": "reply_to",
            "id": "message_id",
            "text": "text",
            "speaker": "user",
            "meta.toxicity": "toxicity",
            "meta.sever_toxicity": "sever_toxicity",
        }
    )

    df = util.preprocessing.std_format_df(df)
    return df


def is_english(text: str) -> bool:
    """
    Check if a given text is written in English.

    :param text: string to check language of
    :return: True if the text is in English, False otherwise
    """
    try:
        lang, prob = langid.classify(text)
        return lang == "en"
    except Exception:
        return False


if __name__ == "__main__":
    main()
