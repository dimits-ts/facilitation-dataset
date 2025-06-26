from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import datasets
import sklearn.metrics


MAX_LENGTH = 2048
SEED = 42
GRAD_ACC_STEPS = 2
EVAL_STEPS = 3000
EPOCHS = 3
BATCH_SIZE = 4

OUTPUT_DIR = Path("../results_only_head")
LOGS_DIR = Path("../logs")


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.dataset.isin(["wikidisputes", "wikitactics"])]
    df = df.reset_index()
    df.is_moderator = df.is_moderator.astype(int)
    return df


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def df_to_train_val_test_dataset(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    train_df, val_df, test_df = train_validate_test_split(
        df, train_percent=0.8, validate_percent=0.1, seed=SEED
    )

    train_dataset = df_to_dataset(train_df, tokenizer)
    val_dataset = df_to_dataset(val_df, tokenizer)
    test_dataset = df_to_dataset(test_df, tokenizer)

    return train_dataset, val_dataset, test_dataset


def df_to_dataset(df, tokenizer):
    x, y = build_discussion_dataset(df, context_window=2)
    dataset = torch_dataset(x, y, tokenizer)
    return dataset


def tokenize_function(tokenizer, example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,  # For Longformer models
    )


def torch_dataset(x, y, tokenizer):
    dataset = datasets.Dataset.from_dict({"text": x, "label": y})
    dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return dataset


def build_discussion_dataset(df, context_window=3):
    inputs = []
    outputs = []

    # Group by conversation
    for conv_id, group in df.groupby("conv_id"):
        texts = group["text"].tolist()
        users = group["user"].tolist()
        labels = group["is_moderator"].tolist()

        for i in range(len(texts)):
            # Get previous `context_window` comments
            start = max(0, i - context_window)
            context = [
                f"<TURN> User {users[j]} posted: {texts[j]}" for j in range(start, i)
            ]

            current = f"<TURN> User {users[i]} posted: {texts[i]}"

            # Combine context and current
            input_text = " ".join(context + [current])
            inputs.append(input_text)
            outputs.append(labels[i])

    return inputs, outputs


def train_validate_test_split(df, train_percent=0.6, validate_percent=0.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)  # numpy operation
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, preds),
        "f1": sklearn.metrics.f1_score(labels, preds),
    }


def train_model(
    model, tokenizer, train_dat, val_dat, test_dat, freeze_base_model: bool
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
        gradient_accumulation_steps=GRAD_ACC_STEPS,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    results = trainer.evaluate(eval_dataset=test_dat)
    print(results)

    trainer.save_model(OUTPUT_DIR / "best_model")
    tokenizer.save_pretrained(OUTPUT_DIR / "best_model")


def main():
    set_seed(SEED)
    model = transformers.LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
    )
    tokenizer = transformers.LongformerTokenizerFast.from_pretrained(
        "allenai/longformer-base-4096", max_length=MAX_LENGTH
    )

    df = pd.read_csv("../pefk.csv")
    df = preprocess_dataset(df)
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
    )


if __name__ == "__main__":
    main()
