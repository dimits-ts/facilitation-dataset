import requests
import json
import time
import argparse
from typing import Iterable, List, Dict

import pandas as pd
from tqdm.auto import tqdm

DELAY_SECS = 1.1


def save_perspective_results(results: List[Dict], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def get_perspective_scores(texts: Iterable[str], api_key: str) -> List[Dict]:
    url = (
        "https://commentanalyzer.googleapis.com/v1alpha1/"
        f"comments:analyze?key={api_key}"
    )
    headers = {"Content-Type": "application/json"}

    results = []
    for idx, text in tqdm(
        enumerate(texts), total=len(texts), desc="Scoring comments"
    ):
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

            toxicity = result["attributeScores"]["TOXICITY"]["summaryScore"][
                "value"
            ]
            severe_toxicity = result["attributeScores"]["SEVERE_TOXICITY"][
                "summaryScore"
            ]["value"]

            results.append(
                {
                    "text": text,
                    "toxicity": toxicity,
                    "severe_toxicity": severe_toxicity,
                }
            )

        except requests.exceptions.RequestException as e:
            print(f"Error at index {idx}: {e}")
            results.append(
                {
                    "text": text,
                    "toxicity": None,
                    "severe_toxicity": None,
                    "error": str(e),
                }
            )

        time.sleep(DELAY_SECS)

    return results


def main(args):
    # Read API key from file
    try:
        with open(args.api_key_file, "r") as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        print(f"Perspective API key file not found: {args.api_key_file}")
        return

    # Load data
    df = pd.read_csv(args.input_csv)
    df = df[df.dataset.isin(["wikitactics", "wikidisputes"])]  # fix: use list

    # Run scoring
    results = get_perspective_scores(df.text, api_key=api_key)
    save_perspective_results(results, args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring on a CSV of texts."
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
        "--output_file",
        type=str,
        default="perspective_results.jsonl",
        help="Output file for results",
    )

    args = parser.parse_args()
    main(args)
