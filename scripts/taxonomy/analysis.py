import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics

from ..util import io
from ..util import preprocessing


MAX_LABEL_LENGTH_CHARS = 10


def _simplify_label(label: str) -> str:
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


def calculate_llm_performance(
    pefk_df: pd.DataFrame, llm_df: pd.DataFrame, graphs_dir: Path
) -> None:
    fora_df = preprocessing.get_human_df(pefk_df, "fora")
    fora_df = fora_df.drop(
        columns=["fora.Personal story", "fora.Personal experience"]
    )
    fora_mapping = {
        "fora.Expressing Agreement": "fora.Express affirmation",
        "fora.Expressing Appreciation": "fora.Express appreciation",
        "fora.Follow-Up Question": "fora.Follow up question",
        "fora.Modeling Example Response": "fora.Provide example",
        "fora.Making Connections": "fora.Make connections",
        "fora.Specific Invitation to Participate": "fora.Specific invitation",
        "fora.Open Invitation to Participate": "fora.Open invitation",
    }
    results = compute_multilabel_accuracy(
        true_df=fora_df, pred_df=llm_df.rename(columns=fora_mapping)
    )

    print("Per-label stats:\n", results["per_label_stats"])
    print("\nHamming accuracy:", results["hamming_accuracy"])
    plot_multilabel_confusion_matrix(
        cm=results["confusion_matrix"],
        labels=results["labels"],
        taxonomy_name="Fora",
    )
    io.save_plot(graphs_dir / "human_llm_cf_matrix_fora.png")

    whow_df = preprocessing.get_human_df(
        pefk_df[pefk_df.is_moderator == 1], "whow"
    )
    whow_mapping = {
        "whow.Instruction": "whow.instruction",
        "whow.Interpretation": "whow.interpretation",
        "whow.Informational Motive": "whow.informational_motive",
        "whow.Confronting": "whow.confronting",
        "whow.Social Motive": "whow.social_motive",
        "whow.Utility": "whow.utility",
        "whow.Supplement": "whow.supplement",
        "whow.Coordinative Motive": "whow.coordinative_motive",
        "whow.Probing": "whow.probing",
    }

    results = compute_multilabel_accuracy(
        true_df=whow_df, pred_df=llm_df.rename(columns=whow_mapping)
    )

    print("Per-label stats:\n", results["per_label_stats"])
    print("\nHamming accuracy:", results["hamming_accuracy"])
    plot_multilabel_confusion_matrix(
        cm=results["confusion_matrix"],
        labels=results["labels"],
        taxonomy_name="WHoW",
    )
    io.save_plot(graphs_dir / "human_llm_cf_matrix_whow.png")


def plot_classifier_results(df: pd.DataFrame, graphs_dir: Path) -> None:
    df.label = df.label.replace(
        {"micro_avg": "Overall (micro)", "macro_avg": "Overall (macro)"}
    )

    # group tactics by taxonomy
    df["prefix"] = df.label.str.split(".").str[0]
    df.label = df.label.apply(_simplify_label)
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

    io.save_plot(graphs_dir / "taxonomy_cls_res.png")


def get_llm_annotations(label_dir: Path):
    dfs = []
    for file in label_dir.iterdir():
        col_name = file.stem
        df = pd.read_csv(file)
        df = df.rename(columns={"is_match": col_name})
        dfs.append(df[col_name])
    dfs.insert(0, df.message_id)

    full_df = pd.concat(dfs, axis=1)
    return full_df


def compute_multilabel_accuracy(true_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    Compute metrics for multilabel classification between human and
    LLM annotations.

    Args:
        true_df (pd.DataFrame): Ground-truth labels.
            Must include 'message_id' and label columns.
        pred_df (pd.DataFrame): Predicted labels.
            Must include 'message_id' and label columns.

    Returns:
        dict: {
            "per_label_stats: pd.DataFrame,
            "hamming_accuracy": float,
            "confusion_matrix": pd.DataFrame,
            "labels": list[str]
        }
    """

    # Merge on message_id to align rows
    merged = true_df.merge(
        pred_df, on="message_id", suffixes=("_true", "_pred")
    )

    # Identify label columns (exclude message_id)
    true_cols = [c for c in true_df.columns if c != "message_id"]
    pred_cols = [c for c in pred_df.columns if c != "message_id"]

    # Use only common labels
    common_labels = [c for c in true_cols if c in pred_cols]
    if not common_labels:
        raise ValueError(
            "No common label columns found between true_df and pred_df."
        )

    # Compute per-label stats
    per_label_stats = {}
    for col in common_labels:
        y_true_col = merged[f"{col}_true"]
        y_pred_col = merged[f"{col}_pred"]

        # Basic metrics
        precision, recall, f1, support = (
            sklearn.metrics.precision_recall_fscore_support(
                y_true_col, y_pred_col, average="binary", zero_division=0
            )
        )
        accuracy = sklearn.metrics.accuracy_score(y_true_col, y_pred_col)

        # Annotator agreement metrics
        cohen_kappa = sklearn.metrics.cohen_kappa_score(y_true_col, y_pred_col)
        mcc = sklearn.metrics.matthews_corrcoef(y_true_col, y_pred_col)

        per_label_stats[col] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cohen_kappa": cohen_kappa,
            "mcc": mcc,
            "support": y_true_col.sum(),
        }

    per_label_stats = pd.DataFrame(per_label_stats).T

    # Flatten results for Hamming accuracy
    Y_true = merged[[f"{col}_true" for col in common_labels]].values
    Y_pred = merged[[f"{col}_pred" for col in common_labels]].values
    hamming_accuracy = 1 - sklearn.metrics.hamming_loss(Y_true, Y_pred)

    # Multilabel confusion matrix
    conf_matrix = sklearn.metrics.multilabel_confusion_matrix(Y_true, Y_pred)

    return {
        "per_label_stats": per_label_stats,
        "hamming_accuracy": hamming_accuracy,
        "confusion_matrix": conf_matrix,
        "labels": common_labels,
    }


def plot_multilabel_confusion_matrix(
    cm: list[list[int]], labels: list[str], taxonomy_name: str
) -> None:
    """
    Plots a grid of confusion matrices (2x2 each) for multilabel
    classification.
    """
    n_labels = len(labels)
    # remove redundant taxonomy title
    labels = [label.split(".")[-1] for label in labels]
    ncols = 3
    nrows = int(np.ceil(n_labels / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows)
    )
    axes = axes.flatten()

    for idx, (label, mat) in enumerate(zip(labels, cm)):
        tn, fp, fn, tp = mat.ravel()
        matrix = np.array([[tn, fp], [fn, tp]])

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            square=True,
            linewidths=0.5,
        )
        axes[idx].set_title(label, fontsize=12)
        axes[idx].set_xlabel("Model Prediction", fontsize=10)
        axes[idx].set_ylabel("Human Annotation", fontsize=10)
        axes[idx].set_xticklabels(["Absent", "Present"], fontsize=9)
        axes[idx].set_yticklabels(["Absent", "Present"], fontsize=9)

    # Remove unused subplots if labels < grid size
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"LLM vs human annot. conf. matrix for {taxonomy_name}")
    plt.tight_layout()


def main(args):
    res_csv_path = Path(args.res_csv_path)
    label_dir = Path(args.label_dir)
    graphs_dir = Path(args.graphs_dir)
    dataset_path = Path(args.dataset_path)

    if not res_csv_path.is_file():
        raise OSError(f"Error: {res_csv_path} is not a file.") from None
    if not graphs_dir.is_dir():
        raise OSError(f"{graphs_dir} is not a directory.") from None
    if not dataset_path.is_file():
        raise OSError(f"Error: {dataset_path} is not a file.") from None
    if not label_dir.is_dir():
        raise OSError(f"{label_dir} is not a directory.") from None

    print("Running human -> LLM evaluation...")
    pefk_df = io.progress_load_csv(dataset_path)
    llm_df = get_llm_annotations(label_dir)
    calculate_llm_performance(pefk_df, llm_df, graphs_dir)

    print("Running LLM -> Transformer evaluation...")
    res_df = pd.read_csv(res_csv_path, index_col=0)
    plot_classifier_results(df=res_df, graphs_dir=graphs_dir)


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
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Directory where the LLM annotation csv files reside",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="The path to the base PEFK dataset",
    )
    main(args=parser.parse_args())
