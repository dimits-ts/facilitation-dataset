import argparse
from pathlib import Path

import shap
import shap.maskers
import torch
import transformers
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import util.io
import util.classification


NUM_SAMPLES_SHAP = 200

# Adapted from 
# https://markaicode.com/transformers-model-interpretability-lime-shap-tutorial/
class TransformerExplainer:
    """
    Production-ready explanation pipeline for transformer models
    """
    
    def __init__(self, model_dir: Path):
        device = 0 if torch.cuda.is_available() else -1

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_dir, reference_compile=False, attn_implementation="eager"
        ).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)

        masker = shap.maskers.Text(self.tokenizer)
        self.shap_explainer = shap.Explainer(self.predict_proba, masker)
        
    def explain(self, text, method=None):
        """
        Generate explanations.
        """     
        return self.shap_explainer([text])  
    
    def batch_explain(self, texts):
        """
        Explain multiple texts efficiently.
        """
        return self.shap_explainer(texts)
    
    def predict_proba(self, texts) -> np.ndarray:
        """
        Predict classification probabilities for a list of texts, 
        returning probabilities for BOTH classes (N, 2) to satisfy SHAP's slicing.
        """
        # --- Normalize SHAP input (Ensure fix for previous TypeError is included) ---
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, (list, tuple)):
            normalized_texts = []
            for t in texts:
                if isinstance(t, (list, np.ndarray)):
                    t = t[0] if len(t) > 0 else "" 
                normalized_texts.append(str(t)) 
            texts = normalized_texts
        elif isinstance(texts, str):
            texts = [texts]
        else:
            raise ValueError(f"Unsupported input type for texts: {type(texts)}")

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits # Assumed shape (N, 1) for BCE model

            # Calculate P(Class 1) using Sigmoid
            p_positive = torch.sigmoid(logits) 
            
            # Calculate P(Class 0) = 1 - P(Class 1)
            p_negative = 1 - p_positive
            
            # Concatenate to create a tensor of shape (N, 2)
            probabilities = torch.cat((p_negative, p_positive), dim=-1)

        return probabilities.cpu().numpy()

    def _normalize_shap_input(self, texts):
        # --- Normalize SHAP input ---
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        if isinstance(texts, (list, tuple)):
            # Flatten nested single-item lists (SHAP often wraps them)
            texts = [t if isinstance(t, str) else str(t) for t in texts]
        return texts

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
    max_context_turns: int
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
    max_context_turns: int
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


def explain_model(
    model_dir: Path,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    label_column: str,
    graph_dir: Path,
    max_length: int = 512,
    max_context_turns: int = 2,
) -> None:
    """
    Generate SHAP explanations for a sample of the test set and save them as
    HTML. Uses DiscussionDataset to construct text sequences with context.
    """

    print("Building test dataset for SHAP explanation...")
    texts = _get_classification_texts(
        model_dir, 
        test_df, 
        full_df, 
        max_length, 
        label_column, 
        max_context_turns
    )

    print(f"Explaining {len(texts)} examples...")
    explainer = TransformerExplainer(model_dir)
    shap_values = explainer.batch_explain(texts)
    plot_data = shap_values[:, :, 1]

    # --- 1. Global Bar Plot (Average Magnitude) ---
    # Shows the mean absolute impact of each word/feature.
    plt.figure(figsize=(10, 6))
    # Passing the full explanation object allows SHAP to handle the tokenization/data
    shap.plots.bar(plot_data, show=False)
    plt.title("Global Feature Importance (Average Magnitude) for Positive Class")
    plt.tight_layout()
    util.io.save_plot(graph_dir / "shap_global_bar_plot.png")
    plt.close()


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
        stratify_col="is_moderator",  # or "should_intervene"
        train_percent=0.7,
        validate_percent=0.2,
    )
    explain_model(
        model_dir=model_dir / "best_model",
        test_df=test_df.sample(n=NUM_SAMPLES_SHAP, random_state=42),
        full_df=classification_df,
        label_column="is_moderator",  # or "should_intervene"
        graph_dir=graph_dir,
        max_length=8192,
        max_context_turns=4,
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
