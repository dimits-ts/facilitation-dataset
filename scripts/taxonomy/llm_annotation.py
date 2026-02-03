import argparse
import logging
import logging.handlers
import yaml
import sys
from pathlib import Path

import torch
import torch.utils.data
import transformers
import pandas as pd
import numpy as np
import coloredlogs
from tqdm.auto import tqdm

from ..util import io
from ..util import classification


MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
MAX_COMMENT_CTX = 2
NUM_COMMENT_SAMPLE = 4000
NUM_DATLOADER_WORKERS = 4

logger = logging.getLogger(__name__)


class Comment:
    def __init__(self, comment: str, user: str, max_len_chars: int = 800):
        self.user = user
        if len(comment) > max_len_chars:
            self.comment = comment[:max_len_chars] + "[...]"
        else:
            self.comment = comment

    def __str__(self):
        return f"Comment by {self.user}: ``{self.comment}''"


class ContextualizedComment(Comment):
    def __init__(self, comment: str, user: str, context: list[Comment]):
        super().__init__(comment=comment, user=user)
        self.context = context if context is not None else []

    def __str__(self):
        ctx_strs = [f"{c.user}: {c.comment}" for c in self.context]
        context = f"Preceding comments: {ctx_strs}"
        return f"{context}\nComment to be classified:\n{super().__str__()}"


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        df_lookup,
        instructions,
        tactic_name,
        description,
        examples,
        tokenizer,
    ):
        self.df = df
        self.df_lookup = df_lookup
        self.instructions = instructions
        self.tactic_name = tactic_name
        self.description = description
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        message_id = row["message_id"]

        # Build context + comment object
        context_comments = fetch_context_chain_comments(
            row.to_dict(), self.df_lookup, MAX_COMMENT_CTX
        )
        comment_obj = ContextualizedComment(
            comment=row["text"],
            user=row.get("user", "unknown"),
            context=context_comments,
        )

        # Create prompt
        prompt = create_prompt_from_input(
            instructions=self.instructions,
            category_name=self.tactic_name,
            description=self.description,
            examples=self.examples,
            input_str=str(comment_obj),
        )

        # Apply chat template and tokenize
        chat_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(chat_text, return_tensors="pt")
        return {
            "message_id": message_id,
            "encoded": encoded,
            "prompt_user": prompt["user"],
        }


def setup_logging(logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = logs_dir / "taxonomies.log"  # include .log extension

    # Rolling over at local midnight; keep last 7 days
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(logfile_path),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
        utc=False,
        # use local time for rollover; switch to True only
        # if you want UTC-based dates
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    # Apply coloredlogs only to console output
    coloredlogs.install(
        level="INFO",
        logger=root_logger,
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        stream=stream_handler.stream,
    )

    logging.captureWarnings(True)


def create_prompt_from_input(
    instructions: str,
    category_name: str,
    description: str,
    examples: list[str],
    input_str: str,
) -> dict:
    """
    Fill the instruction/template file and return a dict with system and
    user messages.
    """
    template = yaml.safe_load(instructions)  # or json.load if it's a .json
    examples_formatted = "\n".join(f"- {ex.strip()}" for ex in examples)
    system_prompt = template["system"]
    user_prompt = (
        template["user"]
        .replace("{{category_name}}", category_name)
        .replace("{{description}}", description.strip())
        .replace("{{examples}}", examples_formatted)
        .replace("{{input}}", input_str.strip())
    )
    return {"system": system_prompt, "user": user_prompt}


def build_comment_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing comments", leave=False
    ):
        msg_id = row.get("message_id")
        if pd.isna(msg_id):
            continue
        lookup[row["message_id"]] = row.to_dict()
    return lookup


def fetch_context_chain_comments(
    comment_row: dict, lookup: dict, max_depth: int
) -> list[Comment]:
    context = []
    current = comment_row
    depth = 0
    while depth < max_depth:
        parent_id = current.get("reply_to")
        if not parent_id or pd.isna(parent_id):
            break
        parent = lookup.get(parent_id)
        if not parent:
            break
        parent_comment = Comment(
            comment=parent.get("text"), user=parent.get("user", "")
        )
        context.append(parent_comment)
        current = parent
        depth += 1
    # return in chronological order (oldest first)
    return list(reversed(context))


def get_sample(
    full_corpus: pd.DataFrame,
    selected_corpus: pd.DataFrame,
) -> pd.Series:
    categories = full_corpus["dataset"].unique()
    n_per_cat = NUM_COMMENT_SAMPLE // len(categories)  # floor division

    sampled = (
        selected_corpus.groupby("dataset")
        .apply(lambda g: g.sample(n=min(len(g), n_per_cat), random_state=42))
        .reset_index(drop=True)
    )
    logger.info(
        "Selected sample sizes:\n" + str(sampled.dataset.value_counts())
    )
    return sampled["message_id"].dropna().astype(str)


def process_all_taxonomies(
    taxonomies: dict[str, dict],
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    instructions: str,
    output_dir: Path,
    model,
    tokenizer,
) -> None:
    df_lookup = build_comment_lookup(full_corpus)

    for tax_name, taxonomy in tqdm(taxonomies.items(), desc="Taxonomies"):
        process_single_taxonomy(
            tax_name=tax_name,
            taxonomy=taxonomy,
            full_corpus=full_corpus,
            classifiable_ids=classifiable_ids,
            instructions=instructions,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            df_lookup=df_lookup,
        )


def process_single_taxonomy(
    tax_name: str,
    taxonomy: dict,
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    instructions: str,
    df_lookup: dict,
    output_dir: Path,
    model,
    tokenizer,
) -> None:
    for tactic_name, tactic in tqdm(
        taxonomy.items(), desc=f"Taxonomy: {tax_name}"
    ):
        logger.info("Processing taxonomy " + tax_name)
        res_df = process_tactic(
            tax_name=tax_name,
            tactic_name=tactic_name,
            tactic=tactic,
            full_corpus=full_corpus,
            classifiable_ids=classifiable_ids,
            instructions=instructions,
            model=model,
            tokenizer=tokenizer,
            df_lookup=df_lookup,
        )

        num_pos = np.count_nonzero(res_df.is_match)
        logger.info(
            f"Tactic {tactic_name} Positives: {num_pos} "
            f"Negatives: {len(res_df) - num_pos}"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{tax_name.strip()}.{tactic_name.strip()}.csv"
        logger.info(f"Saving results to file {out_file}")
        res_df.to_csv(out_file, index=False)


def process_tactic(
    tax_name: str,
    tactic_name: str,
    tactic: dict,
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    df_lookup: dict,
    instructions: str,
    model,
    tokenizer,
) -> pd.DataFrame:
    logger.info(f"Processing tactic {tactic_name}")

    # Filter and prepare subset
    to_classify = full_corpus[
        full_corpus["message_id"].astype(str).isin(classifiable_ids)
    ].reset_index(drop=True)

    dataset = PromptDataset(
        df=to_classify,
        df_lookup=df_lookup,
        instructions=instructions,
        tactic_name=tactic_name,
        description=tactic.get("description"),
        examples=tactic.get("examples"),
        tokenizer=tokenizer,
    )

    # Use num_workers > 0 to parallelize CPU prompt preparation
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_DATLOADER_WORKERS,
        prefetch_factor=4,
    )

    results = []
    logger.info("Starting inference loop...")
    for batch in tqdm(
        loader, total=len(dataset), desc=f"Tactic: {tactic_name}", leave=False
    ):
        encoded = {
            k: v.squeeze(0).to(model.device)
            for k, v in batch["encoded"].items()
        }
        prompt_user = batch["prompt_user"][0]
        message_id = batch["message_id"][0]

        with torch.inference_mode():
            output = model.generate(
                **encoded, max_new_tokens=10, do_sample=False
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in generated_text:
            assistant_reply = generated_text.split("assistant")[-1].strip()
        else:
            assistant_reply = generated_text.split(prompt_user)[-1].strip()

        results.append(
            {
                "message_id": message_id,
                "is_match": parse_response(assistant_reply),
            }
        )

    return pd.DataFrame(results)


def comment_is_tactic(
    encoded_input, prompt_user: str, tokenizer, model
) -> bool:
    try:
        with torch.inference_mode():
            output = model.generate(
                **encoded_input, max_new_tokens=10, do_sample=False
            )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Attempt to isolate the assistant reply
        if "assistant" in generated_text:
            assistant_reply = generated_text.split("assistant")[-1].strip()
        else:
            assistant_reply = generated_text.split(prompt_user)[-1].strip()

        return parse_response(assistant_reply)

    except Exception as e:
        logger.exception("Error during model inference: %s", e)
        return False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_instructions(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_response(res: str) -> bool:
    res = res.strip().lower()
    if "yes" in res and "no" not in res:
        return True
    elif "no" in res and "yes" not in res:
        return False
    else:
        logger.error(f"Non-standard answer detected: {res}")
        return False


def main(args):
    taxonomy_dict: dict = load_yaml(Path(args.taxonomy_file))
    instructions: str = load_instructions(Path(args.prompt_file))
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)

    setup_logging(logs_dir=logs_dir)
    classification.set_seed(42)

    full_corpus = io.progress_load_csv(args.dataset_file)
    full_corpus.text = full_corpus.text.astype(str)
    full_corpus.dataset = full_corpus.dataset.replace(
        {
            "wikiconv": "wikipedia",
            "wikitactics": "wikipedia",
            "wikidisputes": "wikipedia",
        }
    )
    # TODO: remove this
    full_corpus = full_corpus[full_corpus.dataset.isin(["fora", "whow"])]
    logger.info(
        "Full dataset distribution:\n"
        + str(full_corpus.dataset.value_counts())
    )

    mod_corpus = classification.get_implied_actual_mod_df(
        full_corpus,
        mod_threshold=args.mod_probability_thres,
        mod_probability_file=Path(args.mod_probability_file),
    )
    classifiable_ids = get_sample(
        full_corpus=full_corpus, selected_corpus=mod_corpus
    )
    logger.info(
        f"Selected {len(classifiable_ids)} random comments for annotation."
    )

    # Load model and tokenizer once
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto"
    )
    # do NOT chain these
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    try:
        process_all_taxonomies(
            taxonomy_dict,
            full_corpus=full_corpus,
            classifiable_ids=classifiable_ids,
            instructions=instructions,
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
        )
    except Exception as e:
        logger.critical("Critical error encountered. Exiting... %s", e)
        logger.exception("Traceback:")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Classify forum comments using taxonomy categories and an LLM."
        )
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to the full dataset CSV file.",
    )
    parser.add_argument(
        "--taxonomy_file",
        required=True,
        help="Path to the taxonomy YAML file.",
    )
    parser.add_argument(
        "--prompt_file",
        required=True,
        help="Path to the base prompt text file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Dir to save the output CSV files.",
    )
    parser.add_argument(
        "--logs_dir",
        required=True,
        help="Dir to save execution logs.",
    )
    parser.add_argument(
        "--mod_probability_file",
        required=True,
        help=(
            "Path to the mod probability CSV file "
            "(must contain message_id and mod_probability)."
        ),
    )
    parser.add_argument(
        "--mod_probability_thres",
        required=False,
        type=float,
        default=0.5,
        help=(
            "Probability threshold for a comment to be classified "
            "as moderated"
        ),
    )
    args = parser.parse_args()
    main(args)
