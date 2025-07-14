import numpy as np
import pandas as pd
import torch
import torch.nn
import datasets
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import sklearn.metrics


SEED = 42
MAX_LENGTH = 4096
BATCH_SIZE = 32


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
    df: pd.DataFrame, verbose: bool = False
) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []

    print("Indexing...")
    id_to_row = df.set_index("message_id").to_dict("index")

    _iter = (
        tqdm(df.iterrows(), desc="Formatting dataset", total=len(df))
        if verbose
        else df.iterrows
    )
    for _, row in _iter:
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
