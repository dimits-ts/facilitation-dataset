from pathlib import Path
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import util.io


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def comments_per_discussion_plot(df: pd.DataFrame, graph_dir: Path) -> None:
    disc_sizes = (
        df.groupby("conv_id").size().reset_index(name="comments_per_disc")
    )

    # Cap at 40 because there is a tail going up to 1200 comments
    disc_sizes["comments_per_disc"] = disc_sizes["comments_per_disc"].clip(
        upper=40
    )

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=disc_sizes,
        x="comments_per_disc",
        bins=40,
        multiple="stack",
        common_norm=True,
    )
    plt.title("Distribution of Comments per Discussion")
    plt.xlabel("Number of Comments (capped at 40)")
    plt.ylabel("Number of Discussions")
    plt.tight_layout()

    util.io.save_plot(graph_dir / "analysis_comments_per_discussion.png")


def moderation_plot(df: pd.DataFrame, graph_dir: Path) -> float:
    moderator_percent = (
        df.groupby("dataset")["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    moderator_percent["moderator_percent"] *= 100
    order = moderator_percent.sort_values(
        "moderator_percent", ascending=False
    )["dataset"].tolist()

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=moderator_percent,
        x="dataset",
        y="moderator_percent",
        color="steelblue",
        order=order,
    )
    plt.title("Percentage of Moderator Comments per Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    util.io.save_plot(graph_dir / "analysis_moderation_perc.png")


def main(args):
    csv_path = Path(args.dataset_path)
    graph_dir = Path(args.graph_dir)

    print("Loading dataset to extract statistics...")
    df = util.io.progress_load_csv(csv_path)

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

    comments_per_discussion_plot(df, graph_dir)
    comments_per_discussion_plot(df, graph_dir)
    moderation_plot(df, graph_dir=graph_dir)


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
