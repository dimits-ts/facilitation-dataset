import argparse
import json
import queue
import threading
import time
import ast
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

import util.io


DELAY_SECS = 1.1
BATCH_SIZE = 100


def get_perspective_scores(
    df: pd.DataFrame, api_key: str, out_path: Path
) -> None:
    url = (
        "https://commentanalyzer.googleapis.com/v1alpha1/"
        f"comments:analyze?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}

    write_q = queue.Queue()
    writer_thread = threading.Thread(
        target=util.io.writer_thread_func, args=(write_q, out_path)
    )
    writer_thread.start()

    batch = []

    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Scoring comments"
    ):
        text = row["text"]
        msg_id = row["message_id"]

        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}},
            "doNotStore": True,
        }

        try:
            response = requests.post(
                url, headers=headers, data=json.dumps(data)
            )
            response.raise_for_status()
            result = response.json()

            toxicity = _get_response(result, "TOXICITY")
            severe_toxicity = _get_response(result, "SEVERE_TOXICITY")

            result_row = {
                "message_id": msg_id,
                "toxicity": toxicity,
                "severe_toxicity": severe_toxicity,
                "error": None,
            }

        except requests.exceptions.RequestException as e:
            result_row = {
                "message_id": msg_id,
                "toxicity": None,
                "severe_toxicity": None,
                "error": str(e),
            }

        batch.append(result_row)

        if len(batch) >= BATCH_SIZE:
            write_q.put(pd.DataFrame(batch))
            batch = []

        time.sleep(DELAY_SECS)

    if batch:
        write_q.put(pd.DataFrame(batch))

    write_q.put(None)
    writer_thread.join()


def get_ready_df(pefk_df: pd.DataFrame) -> pd.DataFrame:
    pefk_df = pefk_df.loc[
        pefk_df.dataset.isin(["wikiconv", "wikidisputes"]),
        ["message_id", "notes"],
    ]
    tqdm.pandas(desc="Parsing toxicity annotations")
    pefk_df.notes = pefk_df.notes.progress_apply(_safe_parse)
    notes_df = pd.json_normalize(pefk_df.notes)
    pefk_df = pefk_df.drop(columns=["notes"])
    ready_df = pd.concat([pefk_df, notes_df], axis=1)
    ready_df = ready_df.rename(
        columns={
            "meta.toxicity": "toxicity",
            "meta.sever_toxicity": " severe_toxicity",
        }
    )
    ready_df = ready_df.dropna()

    return ready_df


def _get_response(raw_res: dict, attribute: str):
    return raw_res["attributeScores"][attribute]["summaryScore"]["value"]


def _safe_parse(note):
    if not isinstance(note, str):
        print(f"Skipping non-str note: {note} ({type(note)})")
        return {}
    try:
        return ast.literal_eval(note)
    except Exception as e:
        print(f"Failed to parse note: {note} ({e})")
        return {}


def main(args):
    output_path = Path(args.output_path)
    try:
        with open(args.api_key_file, "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        print(f"Perspective API key file not found: {args.api_key_file}")
        return

    df = util.io.progress_load_csv(args.input_csv)
    df = df.loc[
        # exclude datasets that were already annotated with the perspective API
        ~df.dataset.isin(["wikidisputes", "wikiconv"])[
            "message_id", "text", "escalated"
        ],
    ]

    print("Beginning scoring of comments...")
    get_perspective_scores(df, api_key=api_key, out_path=output_path)

    print("Processing existing toxicity annotations")
    ready_df = get_ready_df(df)
    ready_df.to_csv(output_path, mode="a", header=False, index=False)

    print("Finished toxicity annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring and save results to CSV."
    )
    parser.add_argument(
        "--api_key_file",
        type=str,
        required=True,
        help="Path to file containing Perspective API key",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="../pefk.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="perspective_results.csv",
        help="Output path for CSV file",
    )

    args = parser.parse_args()
    main(args)
