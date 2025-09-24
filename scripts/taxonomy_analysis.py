import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import util.io


MAX_LABEL_LENGTH_CHARS = 10


def simplify_label(label: str) -> str:
    if len(label.split(".")) == 1 or label.startswith("Overall"):
        return label

    # Split into prefix part and label part
    prefix_part, label_part = label.split(".", 1)

    # Take only the first prefix (before first underscore)
    prefix = prefix_part.split("_")[0]

    # Take the first two words from the label part
    words = label_part.strip().split()
    suffix = "_".join(words[:2])  # join with underscores

    return f"{prefix}.{suffix}"


def plot_results(df: pd.DataFrame, graphs_dir: Path) -> None:
    df.label = df.label.replace({
        "micro_avg": "Overall (micro)",
        "macro_avg": "Overall (macro)"
    })

    # group tactics by taxonomy
    df["prefix"] = df.label.str.split(".").str[0]
    df.label = df.label.apply(simplify_label)
    df_sorted = df.sort_values(["prefix", "label"]).set_index("label")

    df_heatmap = df_sorted.drop(columns="prefix")
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Metric Value"},
    )

    plt.title(
        "Classifier test performance on LLM-annotated tactics",
        fontsize=14,
        pad=12,
    )
    plt.xlabel("Metric", fontsize=12)
    plt.ylabel("Tactic", fontsize=12)
    plt.tight_layout()

    util.io.save_plot(graphs_dir / "taxonomy_cls_res.png")


def main(args):
    res_csv_path = Path(args.res_csv_path)
    graphs_dir = Path(args.graphs_dir)

    if not res_csv_path.is_file():
        raise OSError(f"Error: {res_csv_path} is not a file.") from None
    if not graphs_dir.is_dir():
        raise OSError(f"{graphs_dir} is not a directory.") from None

    res_df = pd.read_csv(res_csv_path, index_col=0)
    plot_results(df=res_df, graphs_dir=graphs_dir)


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
