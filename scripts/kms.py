from pathlib import Path
import argparse
import joblib

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import sklearn.metrics
import xgboost as xgb
import sentence_transformers
import torch

import util.io
import util.preprocessing
import util.classification


EPOCHS = 200000
EARLY_STOP_ROUNDS = 10
CTX_LENGTH_COMMENTS = 3
N_JOBS = 32
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def get_label_columns(df: pd.DataFrame) -> list[str]:
    return [x for x in df.columns if "." in x]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embed_texts(texts, tokenizer, model, batch_size=8, device="cuda"):
    model.to(device)
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch_texts = texts[i: i + batch_size]

        # Tokenize on CPU
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        # Move only required tensors to CUDA
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = model(**encoded)
            batch_embeds = mean_pooling(
                model_output, encoded["attention_mask"]
            )
            batch_embeds = torch.nn.functional.normalize(
                batch_embeds, p=2, dim=1
            )

        # Move results back to CPU immediately
        all_embeddings.append(batch_embeds.cpu().numpy())

        # Clear CUDA cache to free memory
        del encoded, model_output, batch_embeds
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)


def vectorize_text(train_texts, val_texts, test_texts=None, batch_size=32):
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = sentence_transformers.SentenceTransformer(EMBED_MODEL_NAME)

    print("Embedding training texts...")
    X_train = model.encode(
        train_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Embedding validation texts...")
    X_val = model.encode(
        val_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    X_test = None
    if test_texts is not None:
        print("Embedding test texts...")
        X_test = model.encode(
            test_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    return X_train, X_val, X_test


def train_xgb_models(
    X_train, y_train, X_val, y_val, label_names, output_dir: Path
):
    models = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Training XGBoost models per label...")

    for i, label in enumerate(tqdm(label_names)):
        pos_weight = (len(y_train) - y_train[:, i].sum()) / y_train[:, i].sum()
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            n_estimators=EPOCHS,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=util.classification.SEED,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            n_jobs=N_JOBS,
        )
        model.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        joblib.dump(model, output_dir / f"{label}.joblib")
        models[label] = model

    return models


def evaluate_xgb_models(models, X_test, y_test, label_names):
    print("Evaluating models...")
    preds = []

    for i, label in enumerate(label_names):
        model = models[label]
        pred_proba = model.predict_proba(X_test)[:, 1]
        pred = (pred_proba > 0.5).astype(int)
        preds.append(pred)

    preds = np.vstack(preds).T
    labels = y_test

    results = []
    for i, name in enumerate(label_names):
        results.append(
            {
                "label": name,
                "accuracy": sklearn.metrics.accuracy_score(
                    labels[:, i], preds[:, i]
                ),
                "precision": sklearn.metrics.precision_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
                "recall": sklearn.metrics.recall_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
                "f1": sklearn.metrics.f1_score(
                    labels[:, i], preds[:, i], zero_division=0
                ),
            }
        )

    results.append(
        {
            "label": "micro_avg",
            "accuracy": 1 - sklearn.metrics.hamming_loss(labels, preds),
            "precision": sklearn.metrics.precision_score(
                labels, preds, average="micro", zero_division=0
            ),
            "recall": sklearn.metrics.recall_score(
                labels, preds, average="micro", zero_division=0
            ),
            "f1": sklearn.metrics.f1_score(
                labels, preds, average="micro", zero_division=0
            ),
        }
    )
    results.append(
        {
            "label": "macro_avg",
            "accuracy": 1 - sklearn.metrics.hamming_loss(labels, preds),
            "precision": sklearn.metrics.precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall": sklearn.metrics.recall_score(
                labels, preds, average="macro", zero_division=0
            ),
            "f1": sklearn.metrics.f1_score(
                labels, preds, average="macro", zero_division=0
            ),
        }
    )

    return pd.DataFrame(results)


def train_pipeline(target_df: pd.DataFrame, output_dir: Path, logs_dir: Path):
    label_names = get_label_columns(target_df)
    print("Target label names:", label_names)

    train_df, val_df, test_df = util.classification.train_validate_test_split(
        target_df, train_percent=0.7, validate_percent=0.2
    )

    X_train, X_val, X_test = vectorize_text(
        train_df["text"].tolist(),
        val_df["text"].tolist(),
        test_df["text"].tolist(),
    )

    y_train = train_df[label_names].values
    y_val = val_df[label_names].values
    y_test = test_df[label_names].values

    models = train_xgb_models(
        X_train, y_train, X_val, y_val, label_names, output_dir
    )

    print("Evaluating models...")
    res_df = evaluate_xgb_models(models, X_test, y_test, label_names)
    res_df.to_csv(logs_dir / "res_xgboost.csv", index=False)
    print("Results saved to", logs_dir / "res_xgboost.csv")
    print(res_df)


def main(args):
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)

    util.classification.set_seed(util.classification.SEED)

    df = util.io.progress_load_csv(dataset_path)
    df = util.classification.preprocess_dataset(df)
    mod_df = df[df.is_moderator == 1]
    target_df = util.preprocessing.get_human_df(mod_df, args.sub_dataset_name)
    print(f"Selected {len(target_df)} moderator comments for training.")

    print("Taxonomy distribution:")
    print((target_df[get_label_columns(target_df)] != 0).sum())

    train_pipeline(
        target_df=target_df, output_dir=output_dir, logs_dir=logs_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost multi-label classifier with Qwen embeddings"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to main dataset CSV",
    )
    parser.add_argument(
        "--sub_dataset_name",
        type=str,
        required=True,
        help="Sub-dataset name ('fora' or 'whow')",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--logs_dir", type=str, required=True, help="Logs directory"
    )
    args = parser.parse_args()
    main(args)
