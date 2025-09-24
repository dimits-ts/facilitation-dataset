import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.io


def main(args):
    res_csv_path = Path(args.res_csv_path)
    graphs_dir = Path(args.graphs_dir)

    if not res_csv_path.is_file():
        raise OSError(f"Error: {res_csv_path} is not a file.") from None
    if not graphs_dir.is_dir():
        raise OSError(f"{graphs_dir} is not a directory.") from None

    res_df = pd.read_csv(res_csv_path, index_col=0)
    # Set 'label' as index so it shows up as rows in the heatmap
    df_heatmap = res_df.set_index("label")

    # Create the heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        df_heatmap,
        annot=True,  # show the values
        fmt=".3f",  # 3 decimal places
        cmap="YlGnBu",  # nice color palette
        linewidths=0.5,  # grid lines
        cbar_kws={"label": "Metric Value"},  # colorbar label
    )

    plt.title("Model Performance Metrics per Label", fontsize=14, pad=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Label", fontsize=12)
    plt.tight_layout()

    util.io.save_plot(graphs_dir / "taxonomy_cls_res.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze taxonomy classifier results"
    )
    parser.add_argument(
        "--res_csv_path",
        type=str,
        required=True,
        help="Path to the CSV containing the training results",
    )
    parser.add_argument(
        "--graphs_dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    main(args=parser.parse_args())
