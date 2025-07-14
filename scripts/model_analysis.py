from pathlib import Path
import argparse

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import transformers

import util.classification


DATASET_PATH = Path("pefk.csv")


def load_test_dataset(
    dataset_path: Path, dataset_ls, tokenizer, seed, max_length
):
    df = pd.read_csv(dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls=dataset_ls)
    _, _, test_dataset = util.classification.train_val_test_df(
        df, tokenizer, seed=seed, max_length=max_length
    )
    return test_dataset


def evaluate_model(model, test_dataset, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running model on", device)

    model.eval()
    model.to(device)

    data_collator = transformers.DataCollatorWithPadding(tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=util.classification.BATCH_SIZE,
        collate_fn=data_collator,
    )

    logits_list = []
    labels_list = []
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.detach().cpu().numpy()
            labels = labels.cpu().numpy()

        logits_list.append(logits)
        labels_list.append(labels)

    logits = np.concatenate(logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return util.classification.compute_metrics((logits, labels))


def main(args):
    dataset_ls = args.datasets.split(",")
    model_dir = Path(args.model_dir)
    print(f"Testing model in directory '{model_dir}'")

    util.classification.set_seed(util.classification.SEED)
    model, tokenizer = util.classification.load_trained_model_tokenizer(
        model_dir
    )
    test_dataset = load_test_dataset(
        DATASET_PATH,
        dataset_ls=dataset_ls,
        tokenizer=tokenizer,
        seed=util.classification.SEED,
        max_length=util.classification.MAX_LENGTH,
    )

    metrics = evaluate_model(model, test_dataset, tokenizer)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset selection")
    parser.add_argument(
        "datasets", type=str, help="Comma-separated list of datasets"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Checkpoint directory for trained model",
        required=True,
    )
    args = parser.parse_args()
    main(args)
