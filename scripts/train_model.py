from pathlib import Path
import collections

import numpy as np
import pandas as pd
import torch
import torch.nn
import transformers
import datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split


MAX_LENGTH = 2048
SEED = 42
GRAD_ACC_STEPS = 1
EVAL_STEPS = 150
EPOCHS = 1000
BATCH_SIZE = 64
EARLY_STOP_WARMUP = 5000
EARLY_STOP_THRESHOLD = 10e-5
EARLY_STOP_PATIENCE = 12
COMMENT_WINDOW = 1

OUTPUT_DIR = Path("../results")
LOGS_DIR = Path("../logs")
MODEL_DIR = Path(OUTPUT_DIR / "best_model")


class WeightedLossTrainer(transformers.Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = torch.tensor([pos_weight])

    def compute_loss(self, model, inputs, return_outputs=False):
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


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.dataset == "wikitactics"]
    df = df.reset_index()
    df.is_moderator = df.is_moderator.astype(float)
    return df


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def df_to_train_val_test_dataset(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    train_df, val_df, test_df = train_validate_test_split(
        df,
        stratify_col="is_moderator",
        train_percent=0.75,
        validate_percent=0.15,
        seed=SEED,
    )

    train_dataset = df_to_dataset(train_df, tokenizer)
    val_dataset = df_to_dataset(val_df, tokenizer)
    test_dataset = df_to_dataset(test_df, tokenizer)

    return train_dataset, val_dataset, test_dataset


def df_to_dataset(df, tokenizer):
    x, y = build_discussion_dataset(df, context_window=COMMENT_WINDOW)
    dataset = torch_dataset(x, y, tokenizer)
    return dataset


def tokenize_function(tokenizer, example):
    return tokenizer(
        example["text"],
        padding="max_length",  # needed for batches
        truncation="do_not_truncate",
        max_length=MAX_LENGTH,
    )


def torch_dataset(x, y, tokenizer):
    # Ensure labels are float and shaped (N, 1)
    y = [float(label) for label in y]  # convert to float
    dataset = datasets.Dataset.from_dict({"text": x, "labels": y})
    dataset = dataset.map(
        lambda x: tokenize_function(tokenizer, x), batched=True
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    dataset = dataset.map(reshape_labels)
    return dataset


def reshape_labels(example):
    example["labels"] = torch.tensor([example["labels"]], dtype=torch.float32)
    return example


def build_discussion_dataset(
    df: pd.DataFrame, context_window: int
) -> tuple[list[str], list[str]]:
    inputs = []
    outputs = []

    # Index messages by their ID for quick lookup
    id_to_row = df.set_index("message_id").to_dict("index")

    for conv_id, group in df.groupby("conv_id"):
        for _, row in group.iterrows():
            context = []
            current_id = row["reply_to"]
            steps = 0

            # Follow reply chain backwards up to context_window steps
            while (
                pd.notna(current_id)
                and current_id in id_to_row
                and steps < context_window
            ):
                prev_row = id_to_row[current_id]
                context.append(
                    f"<TURN> User {prev_row['user']} posted: {prev_row['text']}"
                )
                current_id = prev_row["reply_to"]
                steps += 1

            context = context[::-1]  # reverse to get correct order
            current = f"<TURN> User {row['user']} posted: {row['text']}"
            input_text = " ".join(context + [current])
            inputs.append(input_text)
            outputs.append(row["is_moderator"])

    return inputs, outputs


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)  # numpy operation
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds),
    }


def train_model(
    model,
    tokenizer,
    train_dat,
    val_dat,
    test_dat,
    freeze_base_model: bool,
    pos_weight: float,
):
    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        eval_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        save_strategy="steps",
        save_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        logging_strategy="steps",
        logging_dir=LOGS_DIR,
        logging_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        report_to="tensorboard",
    )

    early_stopping = EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    trainer = WeightedLossTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    checkpoints_exist = MODEL_DIR.is_dir()
    trainer.train(resume_from_checkpoint=checkpoints_exist)

    results = trainer.evaluate(eval_dataset=test_dat)
    print(results)

    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


def load_model_tokenizer():
    return _load_default_model_tokenizer()


def _load_default_model_tokenizer():
    model = transformers.LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
        num_labels=1,
        problem_type="multi_label_classification",
    )
    tokenizer = transformers.LongformerTokenizerFast.from_pretrained(
        "allenai/longformer-base-4096", max_length=MAX_LENGTH
    )
    return model, tokenizer


# use this for inference later on
def _load_local_model_tokenizer():
    model = transformers.LongformerForSequenceClassification.from_pretrained(
        MODEL_DIR
    )
    tokenizer = transformers.LongformerTokenizerFast.from_pretrained(MODEL_DIR)
    return model, tokenizer


def main():
    set_seed(SEED)
    model, tokenizer = load_model_tokenizer()

    df = pd.read_csv("../pefk.csv")
    df = preprocess_dataset(df)
    pos_weight = df.is_moderator.sum() / df.shape[0]

    train_dataset, val_dataset, test_dataset = df_to_train_val_test_dataset(
        df, tokenizer
    )

    train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        test_dataset,
        freeze_base_model=True,
        pos_weight=pos_weight,
    )


if __name__ == "__main__":
    main()
