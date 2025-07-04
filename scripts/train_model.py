from pathlib import Path
import argparse

import pandas as pd
import transformers
import sklearn.metrics

import util.classification


MAX_LENGTH = 4096
SEED = 42
GRAD_ACC_STEPS = 1
EVAL_STEPS = 400
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOP_WARMUP = 2000
EARLY_STOP_THRESHOLD = 10e-5
EARLY_STOP_PATIENCE = 20
FINETUNE_ONLY_HEAD = True


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits.reshape(-1) > 0).astype(int)
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
    output_dir: Path,
    logs_dir: Path,
):
    finetuned_model_dir = output_dir / "best_model"
    if freeze_base_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="steps",
        eval_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        save_strategy="steps",
        save_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        logging_strategy="steps",
        logging_dir=logs_dir,
        logging_steps=int(EVAL_STEPS / GRAD_ACC_STEPS),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        report_to="tensorboard",
    )

    early_stopping = util.classification.EarlyStoppingWithWarmupStepsCallback(
        warmup_steps=EARLY_STOP_WARMUP,
        patience=EARLY_STOP_PATIENCE,
        metric_name="eval_loss",
        greater_is_better=False,
    )

    data_collator = transformers.DataCollatorWithPadding(
        tokenizer, padding=True
    )

    trainer = util.classification.WeightedLossTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
        **{"data_collator": data_collator}
    )

    checkpoints_exist = finetuned_model_dir.is_dir()
    trainer.train(resume_from_checkpoint=checkpoints_exist)

    results = trainer.evaluate(eval_dataset=test_dat)
    print(results)

    trainer.save_model(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)


def load_model_tokenizer():
    model = transformers.LongformerForSequenceClassification.from_pretrained(
        "allenai/longformer-base-4096",
        num_labels=1,
        problem_type="multi_label_classification",
    )
    tokenizer = transformers.LongformerTokenizerFast.from_pretrained(
        "allenai/longformer-base-4096", max_length=MAX_LENGTH
    )
    return model, tokenizer


def main(args) -> None:
    dataset_ls = args.dataset.split(",")
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    print("Starting training with datasets: ", dataset_ls)
    util.classification.set_seed(SEED)
    model, tokenizer = load_model_tokenizer()

    df = pd.read_csv("../pefk.csv")
    df = util.classification.preprocess_dataset(df, dataset_ls)
    pos_weight = (df.is_moderator == 0).sum() / (df.is_moderator == 1).sum()

    train_dataset, val_dataset, test_dataset = (
        util.classification.df_to_train_val_test_dataset(
            df, tokenizer, seed=SEED, max_length=MAX_LENGTH
        )
    )

    train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        test_dataset,
        freeze_base_model=FINETUNE_ONLY_HEAD,
        pos_weight=pos_weight,
        output_dir=output_dir,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "dataset", type=str, help="Comma-separated list of datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
        required=True,
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        help="Directory for training logs",
        required=True,
    )
    args = parser.parse_args()
    main(args)
