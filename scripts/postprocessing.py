from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

from tasks import preprocessing_util


INPUT_DIR = Path("../datasets")
OUTPUT_PATH = Path("../pefk.csv")
CHUNK_SIZE = 100_000


def main():
    first_chunk = True  # Track whether to write header

    for file_path in INPUT_DIR.iterdir():
        print(f"Processing {file_path.name}...")
        if file_path.suffix == ".csv":
            try:
                for chunk in tqdm(
                    pd.read_csv(file_path, chunksize=CHUNK_SIZE),
                    total=preprocessing_util.get_num_chunks(
                        file_path=file_path, chunk_size=CHUNK_SIZE
                    ),
                    desc="Exporting"
                ):
                    chunk.to_csv(
                        OUTPUT_PATH,
                        mode="w" if first_chunk else "a",
                        index=False,
                        header=first_chunk,
                    )
                    first_chunk = False
                print(f"{file_path.name} exported to unified dataset.")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
