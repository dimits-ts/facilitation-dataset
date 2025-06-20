from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

INPUT_DIR = Path("../datasets")
OUTPUT_PATH = Path("../pefk.csv")
CHUNK_SIZE = 100_000


def main():
    first_chunk = True  # Track whether to write header

    for file_path in tqdm(
        list(INPUT_DIR.iterdir()), desc="Processing CSV files"
    ):
        if file_path.suffix == ".csv":
            try:
                chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE)

                for chunk in tqdm(chunk_iter, list=):
                    chunk.to_csv(
                        OUTPUT_PATH,
                        mode="w" if first_chunk else "a",
                        index=False,
                        header=first_chunk,
                    )
                    first_chunk = (
                        False  # After first write, stop writing header
                    )
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
