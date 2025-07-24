import numpy as np
import pandas as pd
import torch
import torch.nn
import sklearn.model_selection
import sklearn.metrics
import transformers
from tqdm.auto import tqdm

SEED = 42


class DiscussionDataset(torch.utils.data.Dataset):
    """
    A dataset class that dynamically creates sequences of target comment and N
    previous comments as context. The resulting strings are in XML format where
    <CTX> is a context comment, <USR> is the username, and <TGT> is the target
    comment. Comments are added as context as long as the string would not be
    truncated, preventing truncation of tags and target comment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        max_length: int,
        label_column: str,
        max_context_turns: int = 4,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_column = label_column
        self.max_context_turns = max_context_turns

        self._id2row = self.df.set_index("message_id").to_dict("index")
        self._texts = self.df["text"].tolist()
        self._reply_to = self.df["reply_to"].tolist()
        self._message_ids = self.df["message_id"].tolist()
        self._labels = self.df[self.label_column].astype(float).tolist()

        self._lengths = []
        for idx in tqdm(
            range(len(self.df)), desc="Estimating lengths", total=len(self.df)
        ):
            char_len = self._heuristic_char_length(idx)
            self._lengths.append(char_len)

    def _heuristic_char_length(self, idx: int) -> int:
        """
        Returns an estimated character length of the context + target sequence
        without tokenization. Includes tag and user string overhead.
        """
        total_chars = 0
        turns = 0

        target_row = self.df.iloc[idx]
        tgt = f"<TGT> <USR>{target_row['user']}</USR>{target_row['text']} </TGT>"
        total_chars += len(tgt)

        current_id = target_row["reply_to"]
        while pd.notna(current_id) and turns < self.max_context_turns:
            row = self._id2row.get(current_id)
            if not row:
                break
            ctx = f"<CTX> <USR>{row['user']}</USR> {row['text']} </CTX>"
            total_chars += len(ctx)
            current_id = row["reply_to"]
            turns += 1

        return total_chars

    def _build_sequence(self, idx: int) -> str:
        """
        Dynamically generates the longest possible sequence of context
        comments, up to the specified max_content_turns.
        This means the actual target comment is always included,
        and we avoid non-closing tags during truncation.

        :param idx: _description_
        :type idx: int
        :return: _description_
        :rtype: str
        """
        target_row = self.df.iloc[idx]
        target = (
            f"<TGT> <USR>{target_row['user']}</USR>"
            f"{target_row['text']} </TGT>"
        )

        # Start with just the target and check its length
        encoded = self.tokenizer.encode(
            target,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
        )
        remaining_tokens = self.max_length - len(encoded)
        if remaining_tokens <= 0:
            return target  # Target alone fills or exceeds the limit

        # Now add context incrementally from most recent to oldest
        context = []
        current_id = target_row["reply_to"]
        turns = 0

        while (
            pd.notna(current_id)
            and turns < self.max_context_turns
            and remaining_tokens > 0
        ):
            row = self._id2row.get(current_id)
            if not row:
                break

            turn = f"<CTX> <USR>{row['user']}</USR> {row['text']} </CTX>"
            tokenized = self.tokenizer.encode(
                turn,
                add_special_tokens=False,
                truncation=False,
            )

            if len(tokenized) <= remaining_tokens:
                context.insert(0, turn)  # prepend newer context
                remaining_tokens -= len(tokenized)
            else:
                break  # adding this turn would exceed the limit

            current_id = row["reply_to"]
            turns += 1

        return " ".join(context + [target])

    def _tokenized_length(self, idx):
        seq = self._build_sequence(idx)
        return len(
            self.tokenizer.encode(
                seq,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
        )

    def length(self, idx: int) -> int:
        return self._lengths[idx]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self._build_sequence(idx)
        label = float(self.df.at[idx, self.label_column])
        return {"text": text, "label": label}


def preprocess_dataset(
    df: pd.DataFrame, dataset_ls: list[str] | None = None
) -> pd.DataFrame:
    if dataset_ls is not None:
        df = df[df.dataset.isin(dataset_ls)]
    df = df.reset_index()
    df.is_moderator = df.is_moderator.astype(float)
    df.text = df.text.astype(str)
    return df


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_val_test_df(
    df: pd.DataFrame, tokenizer, seed: int, max_length: int
) -> pd.DataFrame:
    train_df, val_df, test_df = _train_validate_test_split(
        df,
        stratify_col="is_moderator",
        train_percent=0.75,
        validate_percent=0.15,
        seed=seed,
    )

    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.reshape(-1) > 0).astype(int)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds),
    }


def _train_validate_test_split(
    df, stratify_col, train_percent=0.6, validate_percent=0.2, seed=None
):
    # First split into train and temp (validate + test)
    train, temp = sklearn.model_selection.train_test_split(
        df,
        stratify=df[stratify_col],
        test_size=1 - train_percent,
        random_state=seed,
    )

    # Then split temp into validate and test
    validate_size = validate_percent / (1 - train_percent)
    validate, test = sklearn.model_selection.train_test_split(
        temp,
        stratify=temp[stratify_col],
        test_size=1 - validate_size,
        random_state=seed,
    )

    return train, validate, test
