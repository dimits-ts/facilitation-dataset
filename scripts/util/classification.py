from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn
import transformers
import datasets
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics


SEED = 42
MAX_LENGTH = 4096
BATCH_SIZE = 32


class WeightedLossTrainer(transformers.Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()
        outputs = model(**inputs)
        logits = outputs.get("logits").view(-1)
        labels = labels.view(-1)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


class EarlyStoppingWithWarmupStepsCallback(transformers.TrainerCallback):
    def __init__(
        self,
        warmup_steps=500,
        patience=3,
        threshold=0.0,
        metric_name="eval_loss",
        greater_is_better=False,
    ):
        self.warmup_steps = warmup_steps
        self.patience = patience
        self.threshold = threshold
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.wait_count = 0

    def on_evaluate(
        self,
        args,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        metrics,
        **kwargs,
    ):
        current_step = state.global_step

        if current_step < self.warmup_steps:
            return control  # still warming up

        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control  # metric not available yet

        if self.best_metric is None:
            self.best_metric = metric_value
            return control

        operator = (
            (lambda a, b: a > b + self.threshold)
            if self.greater_is_better
            else (lambda a, b: a < b - self.threshold)
        )

        if operator(metric_value, self.best_metric):
            self.best_metric = metric_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                control.should_training_stop = True

        return control


def load_trained_model_tokenizer(model_dir: Path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def preprocess_dataset(
    df: pd.DataFrame, dataset_ls: list[str] | None = None
) -> pd.DataFrame:
    if dataset_ls is not None:
        df = df[df.dataset.isin(dataset_ls)]
    df = df.reset_index()
    df.is_moderator = df.is_moderator.astype(float)
    df.text = df.text.astype(str)
    df = df[df.text.apply(len) > 10]
    return df


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def df_to_train_val_test_dataset(
    df: pd.DataFrame, tokenizer, seed: int, max_length: int
) -> pd.DataFrame:
    train_df, val_df, test_df = _train_validate_test_split(
        df,
        stratify_col="is_moderator",
        train_percent=0.75,
        validate_percent=0.15,
        seed=seed,
    )

    train_dataset = _df_to_dataset(train_df, tokenizer, max_length)
    val_dataset = _df_to_dataset(val_df, tokenizer, max_length)
    test_dataset = _df_to_dataset(test_df, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.reshape(-1) > 0).astype(int)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds),
    }


def _build_discussion_dataset(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []

    # Map message_id to row for quick lookup
    id_to_row = df.set_index("message_id").to_dict("index")

    for _, row in tqdm(df.iterrows(), desc="Formatting dataset"):
        current_comment = row["text"]
        reply_to_id = row["reply_to"]

        # Look up the comment being replied to (if any)
        if pd.notna(reply_to_id) and reply_to_id in id_to_row:
            previous_row = id_to_row[reply_to_id]
            context = previous_row["text"]
        else:
            context = ""

        input_text = f"<CONTEXT> {context} <COMMENT> {current_comment}"
        inputs.append(input_text)
        outputs.append(row["is_moderator"])

    return inputs, outputs


def _df_to_dataset(df, tokenizer, max_length: int):
    x, y = _build_discussion_dataset(df)
    dataset = _torch_dataset(x, y, tokenizer, max_length)
    return dataset


def _tokenize_function(tokenizer, example, max_length: int):
    return tokenizer(
        example["text"],
        padding="longest",  # needed for batches
        truncation=True,
        max_length=max_length,
    )


def _torch_dataset(x, y, tokenizer, max_length: int):
    # Ensure labels are float and shaped (N, 1)
    y = [float(label) for label in y]  # convert to float
    dataset = datasets.Dataset.from_dict({"text": x, "labels": y})
    dataset = dataset.map(
        lambda x: _tokenize_function(tokenizer, x, max_length), batched=True
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataset = dataset.map(_reshape_labels)
    return dataset


def _reshape_labels(example):
    example["labels"] = torch.tensor([example["labels"]], dtype=torch.float32)
    return example


def _train_validate_test_split(
    df, stratify_col, train_percent=0.6, validate_percent=0.2, seed=None
):
    # First split into train and temp (validate + test)
    train, temp = train_test_split(
        df,
        stratify=df[stratify_col],
        test_size=1 - train_percent,
        random_state=seed,
    )

    # Then split temp into validate and test
    validate_size = validate_percent / (1 - train_percent)
    validate, test = train_test_split(
        temp,
        stratify=temp[stratify_col],
        test_size=1 - validate_size,
        random_state=seed,
    )

    return train, validate, test
