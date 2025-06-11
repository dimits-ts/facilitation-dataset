from pathlib import Path

import pandas as pd


ROOT_DIR = Path("/wikiconv")

datasets = []
for file_path in ROOT_DIR.rglob("*.jsonl"):
    dataset = pd.read_json(path_or_buf=file_path, lines=True)
    datasets.append(dataset)

print(dataset[0])