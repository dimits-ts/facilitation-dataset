import numpy as np
import pandas as pd
import torch
import torch.nn
from sklearn.model_selection import train_test_split


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


def train_validate_test_split(
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
