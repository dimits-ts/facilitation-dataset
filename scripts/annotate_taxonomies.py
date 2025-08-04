import argparse
from pathlib import Path
import yaml

import transformers
import pandas as pd
from tqdm.auto import tqdm

import util.io

MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  # placeholder
MAX_COMMENT_CTX = 2  # number of parent comments to include in context


class Comment:
    def __init__(self, comment: str, user: str):
        self.user = user
        self.comment = comment
        self.context = []

    def __str__(self):
        return f"Comment by {self.user}: ``{self.comment}''"


class ClassifiableComment(Comment):
    def __init__(self, comment: str, user: str, context: list[Comment]):
        super().__init__(comment=comment, user=user)
        self.context = context

    def __str__(self):
        ctx_strs = [f"{c.user}: {c.comment}" for c in self.context]
        context = f"Preceding comments: {ctx_strs}"
        return f"{context}\nComment to be classified:\n{super().__str__()}"


def create_prompt_from_input(
    instructions: str,
    category_name: str,
    description: str,
    examples: list[str],
    input_str: str,
) -> str:
    """
    Fill the instruction/template file by replacing:
      - {{category_name}}, {{description}}, {{examples}}, {{input}}
    Examples are joined with newlines and prefixed with '- '.
    """
    examples_formatted = "\n".join(f"- {ex.strip()}" for ex in examples)
    prompt = instructions
    prompt = prompt.replace("{{category_name}}", category_name)
    prompt = prompt.replace("{{description}}", description.strip())
    prompt = prompt.replace("{{examples}}", examples_formatted)
    prompt = prompt.replace("{{input}}", input_str.strip())
    return prompt


def build_comment_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing comments", leave=False):
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
        # wrap parent as a Comment
        parent_comment = Comment(
            comment=parent.get("text"), user=parent.get("user", "")
        )
        context.append(parent_comment)
        current = parent
        depth += 1
    return context


def process_data(
    full_corpus: pd.DataFrame,
    mod_threshold: float,
    mod_probability_file: Path,
) -> pd.Series:
    # Load mod probability file and filter for inferred moderator comments
    mod_prob_df = util.io.progress_load_csv(mod_probability_file)
    high_conf_ids = set(
        mod_prob_df.loc[
            mod_prob_df["mod_probabilities"].astype(float) >= mod_threshold,
            "message_id",
        ].dropna()
    )

    # Moderator-supported comments (true moderators)
    mod_comments = full_corpus[
        (full_corpus.get("is_moderator", False))
        & (full_corpus.get("mod_supported", False))
    ]

    # Inferred moderator comments: non-moderators whose message_id is in high_conf_ids
    inferred_mod_comments = full_corpus[
        (~full_corpus.get("is_moderator", False))
        & (full_corpus["message_id"].isin(high_conf_ids))
    ]

    selected = pd.concat(
        [mod_comments, inferred_mod_comments], ignore_index=True
    )
    return selected["message_id"].dropna().astype(str)


def process_all_taxonomies(
    taxonomies: dict[str, dict],
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    instructions: str,
    output_dir: Path,
    generator,
) -> None:
    for tax_name, taxonomy in tqdm(taxonomies.items(), desc="Taxonomies"):
        process_single_taxonomy(
            tax_name=tax_name,
            taxonomy=taxonomy,
            full_corpus=full_corpus,
            classifiable_ids=classifiable_ids,
            instructions=instructions,
            output_dir=output_dir,
            generator=generator,
        )


def process_single_taxonomy(
    tax_name: str,
    taxonomy: dict,
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    instructions: str,
    output_dir: Path,
    generator,
) -> None:
    for tactic_name, tactic in tqdm(
        taxonomy.items(), desc=f"Taxonomy: {tax_name}"
    ):
        process_tactic(
            tax_name=tax_name,
            tactic_name=tactic_name,
            tactic=tactic,
            full_corpus=full_corpus,
            classifiable_ids=classifiable_ids,
            instructions=instructions,
            output_dir=output_dir,
            generator=generator,
        )


def process_tactic(
    tax_name: str,
    tactic_name: str,
    tactic: dict,
    full_corpus: pd.DataFrame,
    classifiable_ids: pd.Series,
    instructions: str,
    output_dir: Path,
    generator,
) -> None:
    description = tactic.get("description")
    examples = tactic.get("examples")

    # Build lookup from full corpus so context chain is intact
    # TODO: delete
    lookup = build_comment_lookup(full_corpus.head(20000))

    # Filter the rows of full_corpus to only those we should classify
    to_classify = full_corpus[
        full_corpus["message_id"].astype(str).isin(set(classifiable_ids))
    ]

    results = []
    for _, row in tqdm(
        to_classify.iterrows(),
        total=len(to_classify),
        leave=False,
        desc=f"Tactic: {tactic_name}",
    ):
        message_id = row.get("message_id")
        context_comments = fetch_context_chain_comments(
            row.to_dict(), lookup, MAX_COMMENT_CTX
        )
        user = row.get("user", "unknown")
        comment_text = row.get("text")
        classifiable = ClassifiableComment(
            comment=comment_text, user=user, context=context_comments
        )

        input_str = str(classifiable)
        prompt = create_prompt_from_input(
            instructions=instructions,
            category_name=tactic_name,
            description=description,
            examples=examples,
            input_str=input_str,
        )
        # TODO: delete
        print(prompt)

        is_match = comment_is_tactic(prompt=prompt, generator=generator)
        results.append({"message_id": message_id, "is_match": is_match})

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{tax_name.strip()}.{tactic_name.strip()}.csv"
    pd.DataFrame(results, columns=["message_id", "is_match"]).to_csv(
        out_file, index=False
    )


def comment_is_tactic(prompt: str, generator) -> bool:
    """
    Uses the transformers pipeline to ask the model. Expects the model to answer with 'yes' or 'no'.
    Parses the first clear yes/no.
    """
    try:
        output = generator(
            prompt,
            max_new_tokens=10,
            do_sample=False,  # deterministic
        )
        # pipeline returns list with generated_text
        res = output[0]["generated_text"][len(prompt) :].strip().lower()
        res = parse_response(res)
        return res
    except Exception as e:
        print("Exception occurred during inference: ", e)
        return False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_instructions(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_response(res: str) -> bool:
    if res.startswith("yes"):
        return True
    if res.startswith("no"):
        return False
    # sometimes model replies like "yes." or "nope" or "no."
    if res.startswith("y"):
        return True
    if res.startswith("n"):
        return False

    print("Non-standard answer detected: {res}")
    return False


def main(args):
    taxonomy_dict: dict = load_yaml(Path(args.taxonomy_file))
    instructions: str = load_instructions(Path(args.prompt_file))
    output_dir = Path(args.output_dir)

    full_corpus = util.io.progress_load_csv(args.dataset_file)
    classifiable_ids = process_data(
        full_corpus,
        mod_threshold=args.mod_probability_thres,
        mod_probability_file=Path(args.mod_probability_file),
    )

    # Load model and tokenizer once
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto"
    )
    generator = transformers.TextGenerationPipeline(
        model=model, tokenizer=tokenizer
    )

    process_all_taxonomies(
        taxonomy_dict,
        full_corpus=full_corpus,
        classifiable_ids=classifiable_ids,
        instructions=instructions,
        output_dir=output_dir,
        generator=generator,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify forum comments using taxonomy categories and an LLM."
    )
    parser.add_argument(
        "--mod_probability_thres",
        required=False,
        type=float,
        default=0.5,
        help="Probability threshold for a comment to be classified as moderated",
    )
    parser.add_argument(
        "--mod_probability_file",
        required=True,
        help="Path to the mod probability CSV file (must contain message_id and mod_probability).",
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
    args = parser.parse_args()
    main(args)
