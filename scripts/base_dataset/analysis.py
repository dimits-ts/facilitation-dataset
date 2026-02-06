
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

from pathlib import Path
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from ..util import io


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def main(args):
    csv_path = Path(args.dataset_path)
    graph_dir = Path(args.graph_dir)

    print("Loading dataset to extract statistics...")
    df = io.progress_load_csv(csv_path)

    print("*" * 25)
    print("Comments per discussion:")
    print(df.groupby("conv_id").size().describe())

    print("*" * 25)
    print("Comments per dataset:")
    print(df.groupby("dataset")["message_id"].nunique())

    print("*" * 25)
    print("Discussions per dataset:")
    print(df.groupby("dataset")["conv_id"].nunique())

    print("*" * 25)
    print("Word count per comment:")
    print(
        df.text.astype(str)
        .apply(lambda x: x.split())
        .apply(len)
        .astype(int)
        .describe()
    )

    print("*" * 25)
    print(f"Dataset total size: {convert_bytes(csv_path.stat().st_size)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    args = parser.parse_args()
    main(args)
