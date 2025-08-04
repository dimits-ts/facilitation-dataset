import argparse
from pathlib import Path
import yaml
import pandas as pd
from tqdm.auto import tqdm

import util.io

MODEL_NAME = "gpt2"  # placeholder
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


def create_prompt(
    instructions: str,
    category_name: str,
    description: str,
    examples: list[str],
    comment: str,
    context: list[str],
) -> str:
    """
    Fill the instruction/template file by replacing:
      - {{category_name}}, {{description}}, {{examples}}
      - <COMMENT>, <CONTEXT>
    Examples are joined with newlines and prefixed with '- '.
    If multiple context comments are provided, they are joined with double newlines.
    """
    # Prepare replacements
    examples_formatted = "\n".join(f"- {ex.strip()}" for ex in examples)
    context_text = "\n\n".join(c.strip() for c in context) if context else ""
    prompt = instructions
    prompt = prompt.replace("{{category_name}}", category_name)
    prompt = prompt.replace("{{description}}", description.strip())
    prompt = prompt.replace("{{examples}}", examples_formatted)
    prompt = prompt.replace("<CONTEXT>", context_text)
    prompt = prompt.replace("<COMMENT>", comment.strip())
    return prompt


def build_comment_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        msg_id = row.get("message_id")
        if pd.isna(msg_id):
            continue
        lookup[row["message_id"]] = row.to_dict()
    return lookup


def fetch_context_chain(
    comment_row: dict, lookup: dict, max_depth: int
) -> list[str]:
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
        context.append(parent.get("message", ""))
        current = parent
        depth += 1
    return context


def process_data(
    data: pd.DataFrame, mod_threshold: float, mod_probability_file: Path
) -> pd.DataFrame:
    # Load mod probability file and filter for inferred moderator comments
    mod_prob_df = util.io.progress_load_csv(mod_probability_file)
    high_conf_ids = set(
        mod_prob_df.loc[
            mod_prob_df["mod_probabilities"].astype(float) >= mod_threshold,
            "message_id",
        ].dropna()
    )

    # Moderator-supported comments (true moderators)
    mod_comments = data[
        (data.get("is_moderator", False)) & (data.get("mod_supported", False))
    ]

    # Inferred moderator comments: non-moderators whose message_id is in high_conf_ids
    inferred_mod_comments = data[
        (~data.get("is_moderator", False))
        & (data["message_id"].isin(high_conf_ids))
    ]

    selected = pd.concat(
        [mod_comments, inferred_mod_comments], ignore_index=True
    )
    return selected


def process_all_taxonomies(
    taxonomies: dict[str, dict],
    data: pd.DataFrame,
    instructions: str,
    output_dir: Path,
) -> None:
    for tax_name, taxonomy in tqdm(taxonomies.items(), desc="All Taxonomies"):
        process_single_taxonomy(
            tax_name, taxonomy, data, instructions, output_dir
        )


def process_single_taxonomy(
    tax_name: str,
    taxonomy: dict,
    data: pd.DataFrame,
    instructions: str,
    output_dir: Path,
) -> None:
    for tactic_name, tactic in tqdm(
        taxonomy.items(), desc=f"Taxonomy: {tax_name}"
    ):
        process_tactic(
            tax_name, tactic_name, tactic, data, instructions, output_dir
        )


def process_tactic(
    tax_name: str,
    tactic_name: str,
    tactic: dict,
    data: pd.DataFrame,
    instructions: str,
    output_dir: Path,
) -> None:
    description = tactic.get("Description", "")
    examples = tactic.get("Examples", [])
    lookup = build_comment_lookup(data)
    results = []
    for _, row in tqdm(
        data.iterrows(),
        total=len(data),
        leave=False,
        desc=f"Processing tactic: {tactic_name}",
    ):
        message_id = row.get("message_id")
        comment_text = row.get("message", "")
        user = row.get("user", "unknown")
        context_texts = fetch_context_chain(
            row.to_dict(), lookup, MAX_COMMENT_CTX
        )
        prompt = create_prompt(
            instructions=instructions,
            category_name=tactic_name,
            description=description,
            examples=examples,
            comment=comment_text,
            context=context_texts,
        )

        is_match = comment_is_tactic(prompt=prompt)
        results.append(
            {
                "message_id": message_id,
                "is_match": is_match,
                "comment": comment_text,
                "user": user,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{tax_name}__{tactic_name}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)


def comment_is_tactic(prompt: str) -> bool:
    # Placeholder heuristic. Replace with real LLM inference.
    lowered = prompt.lower()
    if "harass" in lowered or "attack" in lowered:
        return True
    return False


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_instructions(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(args):
    taxonomy_dict: dict = load_yaml(Path(args.taxonomy_file))
    instructions: str = load_instructions(Path(args.prompt_file))
    output_dir = Path(args.output_dir)

    data = util.io.progress_load_csv(args.dataset_file)
    df = process_data(
        data,
        mod_threshold=args.mod_probability_thres,
        mod_probability_file=Path(args.mod_probability_file),
    )

    process_all_taxonomies(taxonomy_dict, df, instructions, output_dir)


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
