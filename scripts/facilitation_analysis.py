import argparse
from pathlib import Path

import shap
import shap.maskers
import torch
import transformers
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import util.io
import util.classification


NUM_SAMPLES_SHAP = 500
MAX_CONTEXT_TURNS = 2
MAX_LENGTH = 4096
LABEL_COLUMN = "is_moderator"


def collate_fn(tokenizer, batch):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch]).unsqueeze(1)
    enc = tokenizer(
        texts,
        padding="longest",
        truncation=False,
        max_length=8192,
        return_tensors="pt",
    )
    enc["labels"] = labels
    return enc


def _get_classification_texts(
    model_dir: Path,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    max_length: int,
    label_column: str,
    max_context_turns: int,
) -> list[str]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    # Build dataset and take a sample (too many samples make SHAP very slow)
    ds = util.classification.DiscussionDataset(
        target_df=test_df,
        full_df=full_df,
        tokenizer=tokenizer,
        max_length=max_length,
        label_column=label_column,
        max_context_turns=max_context_turns,
    )
    texts = [ds[i]["text"] for i in range(len(ds))]
    return texts


def augmented_moderation_plot(
    df: pd.DataFrame,
    mod_probability_file: Path,
    mod_threshold: float,
    graph_dir: Path,
) -> None:
    moderator_percent = (
        df.groupby("dataset")["is_moderator"]
        .mean()
        .reset_index(name="moderator_percent")
    )
    moderator_percent["moderator_percent"] *= 100

    # --- Inferred moderator percentages ---
    mod_probability_file = Path(mod_probability_file)
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

    util.io.save_plot(graph_dir / "augmented_analysis_moderation_perc.png")
    plt.close()


def _get_explainer(model_dir: Path, max_length: int):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir, reference_compile=False, attn_implementation="eager"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    pipe = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=max_length,
        top_k=None,
    )
    pmodel = shap.models.TransformersPipeline(pipe, rescale_to_logits=True)
    masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
    explainer = shap.Explainer(pmodel, masker)
    return explainer


def explain_model_global(
    explainer,
    texts: list[str],
    graph_dir: Path,
    max_length: int = 512,
) -> None:
    """
    Generate SHAP explanations for a sample of the test set and save them as
    HTML. Uses DiscussionDataset to construct text sequences with context.
    """
    print(f"Explaining {len(texts)} examples...")

    shap_values = explainer(texts)
    plot_data = shap_values[:, :, -1]

    # --- 1. Global Bar Plot (Average Magnitude) ---
    # Shows the mean absolute impact of each word/feature.
    plt.figure(figsize=(10, 6))
    # Passing the full explanation object allows SHAP to handle the
    # tokenization/data
    shap.plots.bar(plot_data, show=False, max_display=20)
    plt.title("Token importance for facilitation detection")
    plt.tight_layout()
    util.io.save_plot(graph_dir / "shap_global_bar_plot.png")
    plt.close()


def explain_model_local(explainer, text: str, graph_dir: Path): ...


def main(args):
    model_dir = Path(args.model_dir)
    graph_dir = Path(args.graph_dir)
    util.classification.set_seed(util.classification.SEED)
    df = util.io.progress_load_csv(Path(args.dataset_path))

    """augmented_moderation_plot(
        df,
        Path(args.mod_probability_file),
        args.mod_probability_thres,
        graph_dir=graph_dir,
    )"""

    print(
        "Running explanation algorithms on model for "
        f"N={NUM_SAMPLES_SHAP} test-set comments..."
    )
    classification_df = util.classification.preprocess_dataset(df)
    _, _, test_df = util.classification.train_validate_test_split(
        classification_df,
        straglobal_tify_col="is_moderator",  # or "should_intervene"
        train_percent=0.7,
        validate_percent=0.2,
    )
    test_df = test_df.sample(n=NUM_SAMPLES_SHAP, random_state=42)

    print("Building explainer for SHAP explanation...")
    explainer = _get_explainer(model_dir=model_dir, max_length=MAX_LENGTH)

    print("Building test dataset for SHAP explanation...")
    texts = _get_classification_texts(
        model_dir=model_dir,
        test_df=test_df,
        full_df=df,
        max_length=MAX_LENGTH,
        label_column=LABEL_COLUMN,
        max_context_turns=MAX_CONTEXT_TURNS,
    )

    explain_model_global(
        explainer=explainer,
        texts=texts,
        graph_dir=graph_dir,
        max_length=MAX_LENGTH,
    )
    print("Done.")

    print("Facilitator analysis concluded.")


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
        "--mod_probability_file",
        required=True,
        help="Path to the mod probability CSV file.",
    )
    parser.add_argument(
        "--mod_probability_thres",
        required=False,
        type=float,
        default=0.6,
        help=(
            "Prob. threshold for a comment to be classified "
            "as a moderator intervention."
        ),
    )
    parser.add_argument(
        "--graph_dir",
        type=str,
        required=True,
        help="Directory where the graphs will be exported to",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Checkpoint directory for trained model.",
    )

    args = parser.parse_args()
    main(args)
