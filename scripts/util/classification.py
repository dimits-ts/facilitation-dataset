import random
from collections.abc import Iterable

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
        tgt = (
            "<TGT>"
            f"<USR>{target_row['user']}</USR>"
            f"{target_row['text']}"
            "</TGT>"
        )
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


class SmartBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """
    Yields lists of indices such that items inside a batch have similar
    sequence length.  That keeps padding - and thus compute - to a minimum.

    • Batches are *shuffled* every epoch, but order inside each batch
      is irrelevant (Trainer will not re-shuffle).
    • Works for any dataset that yields a dict with 'input_ids'.
    """

    def __init__(self, dataset, batch_size, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # just ask the dataset
        self.lengths = [dataset.length(i) for i in range(len(dataset))]
        self.sorted_indices = sorted(
            range(len(dataset)), key=self.lengths.__getitem__
        )

    def __iter__(self):
        # -- bucketed indices, then shuffle buckets --
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        random.shuffle(batches)  # different order every epoch
        for b in batches:
            yield b

    def __len__(self):
        full, rest = divmod(len(self.sorted_indices), self.batch_size)
        return full if (rest == 0 or self.drop_last) else full + 1


class BucketedTrainer(WeightedLossTrainer):
    """
    Overrides get_train_dataloader() and get_eval_dataloader()
    so Trainer uses our bucketed sampler for train and default sampler
    + collate_fn for eval.
    """

    def __init__(self, bucket_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_batch_size = bucket_batch_size

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training dataset must be provided")

        sampler = SmartBucketBatchSampler(
            self.train_dataset,
            batch_size=self.bucket_batch_size,
            drop_last=False,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    def get_eval_dataloader(
        self, eval_dataset=None
    ) -> torch.utils.data.DataLoader:
        eval_dataset = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation dataset must be provided")

        # Use default sequential sampler for eval + your collate_fn
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.bucket_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.reshape(-1) > 0).astype(int)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds),
    }


def train_validate_test_split(
    df,
    train_percent=0.6,
    validate_percent=0.2,
    seed=None,
    stratify_col: str | None = None,
):
    # First split into train and temp (validate + test)
    train, temp = sklearn.model_selection.train_test_split(
        df,
        stratify=None if stratify_col is None else df[stratify_col],
        test_size=1 - train_percent,
        random_state=seed,
    )

    # Then split temp into validate and test
    validate_size = validate_percent / (1 - train_percent)
    validate, test = sklearn.model_selection.train_test_split(
        temp,
        stratify=None if stratify_col is None else temp[stratify_col],
        test_size=1 - validate_size,
        random_state=seed,
    )

    return train, validate, test


def results_to_df(
    individual_raw: str, all_raw: str, test_metrics: Iterable[str]
) -> pd.DataFrame:
    # ── parse keys →   eval_<dataset>_<metric>  ──────────────────────────────
    per_ds: dict[str, dict[str, float]] = {}
    for k, v in individual_raw.items():
        if not k.startswith("eval_"):
            continue
        _, rest = k.split("_", 1)  # drop leading 'eval_'
        # split once from the right so dataset names with '_' are handled
        ds, metric = rest.rsplit("_", 1)
        if metric in test_metrics:
            per_ds.setdefault(ds, {})[metric] = v

    # ── add aggregate (“ALL”) row ────────────────────────────────────────────
    per_ds["ALL"] = {m: all_raw[f"eval_{m}"] for m in test_metrics}

    # ── to DataFrame, ordered columns ────────────────────────────────────────
    df_metrics = (
        pd.DataFrame.from_dict(per_ds, orient="index")
        .reindex(columns=["loss", "accuracy", "f1"])  # fixed order
        .sort_index()
    )

    return df_metrics
