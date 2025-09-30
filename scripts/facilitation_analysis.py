from pathlib import Path
import argparse

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import util.io


INPUT_PATH = Path("../pefk.csv")
GRAPH_PATH = Path("../graphs")


def augmented_moderation_plot(
    df: pd.DataFrame, mod_probability_file: Path, mod_threshold: float
) -> None:
    moderator_percent = (
        df.groupby("dataset")["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    moderator_percent["moderator_percent"] *= 100

    # --- Inferred moderator percentages ---
    mod_probability_file = Path("../output_datasets/pefk_mod.csv")
    mod_threshold = 0.6

    mod_prob_df = util.io.progress_load_csv(mod_probability_file)
    high_conf_ids = set(
        mod_prob_df.loc[
            mod_prob_df.mod_probabilities.astype(float) >= mod_threshold,
            "message_id",
        ].dropna()
    )

    # Count inferred moderators per dataset
    inferred_mod_counts = (
        df.loc[df.message_id.isin(high_conf_ids)]
        .groupby("dataset")["message_id"]
        .count()
        .reset_index(name="inferred_mod_count")
    )

    # Normalize to percentage
    dataset_totals = (
        df.groupby("dataset")["message_id"]
        .count()
        .reset_index(name="total_comments")
    )
    inferred_mod_percent = inferred_mod_counts.merge(
        dataset_totals, on="dataset"
    )
    inferred_mod_percent["inferred_mod_percent"] = (
        inferred_mod_percent["inferred_mod_count"]
        / inferred_mod_percent["total_comments"]
        * 100
    )

    # --- Combine in long format ---
    plot_df = moderator_percent.merge(
        inferred_mod_percent[["dataset", "inferred_mod_percent"]],
        on="dataset",
        how="outer",
    )

    plot_df = plot_df.melt(
        id_vars="dataset",
        value_vars=["moderator_percent", "inferred_mod_percent"],
        var_name="Type",
        value_name="Percentage",
    )

    # --- Plot ---
    order = moderator_percent.sort_values(
        "moderator_percent", ascending=False
    )["dataset"].tolist()

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=plot_df,
        x="dataset",
        y="Percentage",
        hue="Type",
        palette={
            "moderator_percent": "steelblue",
            "inferred_mod_percent": "orange",
        },
        order=order,
    )

    plt.title(
        f"Percentage of actual and inferred (with {mod_threshold * 100:.0f}% "
        "confidence) facilitative comments"
    )
    plt.xlabel("Dataset")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)

    handles, labels = ax.get_legend_handles_labels()
    new_labels = ["True facilitation", "Inferred facilitation"]
    plt.legend(handles, new_labels, title="")
    plt.tight_layout()

    util.io.save_plot(GRAPH_PATH / "augmented_analysis_moderation_perc.png")


def main(args):
    print("Loading dataset...")
    df = util.io.progress_load_csv(INPUT_PATH)

    augmented_moderation_plot(
        df, Path(args.mod_probability_file), args.mod_probability_thres
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate discussion statistics and moderation plots."
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--mod_probability_file",
        required=False,
        help="Path to the mod probability CSV file.",
        default="no.path.given",
    )
    parser.add_argument(
        "--mod_probability_thres",
        required=False,
        type=float,
        default=0.6,
        help="Prob. threshold for a comment to be classified as moderated.",
    )

    args = parser.parse_args()
    main(args)
