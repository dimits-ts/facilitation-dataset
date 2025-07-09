from pathlib import Path
import argparse

from tqdm.auto import tqdm
import pandas as pd
import torch
import transformers
import sklearn.metrics

import util.classification


DATASET_PATH = Path("pefk.csv")
# make sure this is the same as the one used during model training
# in order to obtain the proper test set
SEED = 42
MAX_LENGTH = 4096


def load_model_tokenizer(model_dir: Path):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def load_test_dataset(
    dataset_path: Path, dataset_ls, tokenizer, seed, max_length
):
    df = pd.read_csv(dataset_path)
    df = util.classification.preprocess_dataset(df, dataset_ls=dataset_ls)
    _, _, test_dataset = util.classification.df_to_train_val_test_dataset(
        df, tokenizer, seed=seed, max_length=max_length
    )
    return test_dataset


def evaluate_model(model, test_dataset):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    all_labels = []

    for batch in tqdm(test_dataset):
        # Move to device
        input_ids = batch["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(model.device)
        labels = batch["labels"].unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = sklearn.metrics.accuracy_score(all_labels, all_preds)
    f1 = sklearn.metrics.f1_score(all_labels, all_preds, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def main(args):
    dataset_ls = args.dataset.split(",")
    model_dir = Path(args.model_dir)
    print(f"Testing model in directory '{model_dir}'")

    util.classification.set_seed(SEED)
    results = []

    model, tokenizer = load_model_tokenizer(model_dir)
    test_dataset = load_test_dataset(
        DATASET_PATH,
        dataset_ls=dataset_ls,
        tokenizer=tokenizer,
        seed=SEED,
        max_length=MAX_LENGTH,
    )

    # If test_dataset is a Dataset object, convert to DataLoader
    # for batch eval
    if hasattr(test_dataset, "collate_fn"):
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=8, collate_fn=test_dataset.collate_fn
        )
    else:
        test_loader = test_dataset  # assume iterable of dicts if pre-batched

    metrics = evaluate_model(model, test_loader)
    results.append({"model": str(model_dir), **metrics})

    results_df = pd.DataFrame(results)
    print(results_df)


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
