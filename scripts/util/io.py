import queue
from pathlib import Path

import pandas as pd


def _append_batch_to_csv(
    df_batch: pd.DataFrame, out_path: Path, *, first_batch: bool
) -> None:
    mode = "w" if first_batch else "a"
    df_batch.to_csv(out_path, mode=mode, header=first_batch, index=False)


def writer_thread_func(write_queue: queue.Queue, out_path: Path):
    first_batch = True
    while True:
        df_batch = write_queue.get()
        if df_batch is None:
            break
        _append_batch_to_csv(df_batch, out_path, first_batch=first_batch)
        first_batch = False
        write_queue.task_done()
