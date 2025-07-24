import queue
import math
import subprocess
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def writer_thread_func(write_queue: queue.Queue, out_path: Path):
    first_batch = True
    while True:
        df_batch = write_queue.get()
        if df_batch is None:
            break
        _append_batch_to_csv(df_batch, out_path, first_batch=first_batch)
        first_batch = False
        write_queue.task_done()


def progress_load_csv(
    csv_path: Path | str, chunksize: int = 100000
) -> pd.DataFrame:
    return pd.concat(
        [
            chunk
            for chunk in tqdm(
                pd.read_csv(csv_path, chunksize=chunksize),
                desc="Loading dataset",
                total=get_num_chunks(csv_path, chunksize),
                bar_format="{l_bar}{bar} {percentage:.0f}%",
                leave=False
            )
        ]
    )


def get_num_chunks(file_path: Path, chunk_size: int) -> int:
    result = subprocess.run(
        ["wc", "-l", str(file_path)], capture_output=True, text=True
    )
    return math.ceil(int(result.stdout.strip().split()[0]) / chunk_size)


def _append_batch_to_csv(
    df_batch: pd.DataFrame, out_path: Path, *, first_batch: bool
) -> None:
    mode = "w" if first_batch else "a"
    df_batch.to_csv(out_path, mode=mode, header=first_batch, index=False)
