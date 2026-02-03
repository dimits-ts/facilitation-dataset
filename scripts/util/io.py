
# The PEFK (Prosocial and Effective Facilitation in Konversations) Dataset
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import queue
import math
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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
                leave=False,
            )
        ]
    )


def get_num_chunks(file_path: Path, chunk_size: int) -> int:
    # Is fooled by newlines inside the data, overstimates dataset size
    # However is the only way to obtain an estimate without reading the whole
    # file, which would defeat the point.
    result = subprocess.run(
        ["wc", "-l", str(file_path)], capture_output=True, text=True
    )
    return math.ceil(int(result.stdout.strip().split()[0]) / chunk_size)


def _append_batch_to_csv(
    df_batch: pd.DataFrame, out_path: Path, *, first_batch: bool
) -> None:
    mode = "w" if first_batch else "a"
    df_batch.to_csv(out_path, mode=mode, header=first_batch, index=False)


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path: The full path (including filename)
        where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {path.resolve()}")
