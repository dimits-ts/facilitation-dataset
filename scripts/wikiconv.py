from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


INPUT_DIR = Path("../downloads/wikiconv")
INTERMIDATE_CSV_PATH = Path("wikiconv_combined.csv")
CHUNK_SIZE = 10000


def get_num_chunks(file_path: Path, chunk_size: int) -> int:
    """Count lines in a JSONL file and compute number of chunks."""
    with file_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    return (total_lines + chunk_size - 1) // chunk_size  # Ceiling division


first_chunk = True
jsonl_files = list(INPUT_DIR.rglob("*.jsonl"))
for file_path in tqdm(jsonl_files, desc="Files"):
    for chunk in tqdm(
        pd.read_json(file_path, lines=True, chunksize=CHUNK_SIZE),
        total=get_num_chunks(file_path, CHUNK_SIZE),
        leave=False,
    ):
        chunk.to_csv(
            INTERMIDATE_CSV_PATH, mode="a", index=False, header=first_chunk
        )
        first_chunk = False
