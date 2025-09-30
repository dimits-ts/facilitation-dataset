from pathlib import Path

import pandas as pd

import util.preprocessing


INPUT_DIR = Path("../downloads/whow/data")
OUTPUT_PATH = Path("../datasets/whow.csv")
DIALOGUE_ACTS = [
    "probing",
    "confronting",
    "instruction",
    "interpretation",
    "supplement",
    "utility",
]


def import_df(input_dir: Path) -> pd.DataFrame:
    excel_ls = []
    for excel_file in input_dir.rglob("*.xlsx"):
        df = pd.read_excel(excel_file)
        df["conv_id"] = excel_file.name
        excel_ls.append(df)

    df = pd.concat(excel_ls, ignore_index=True)
    return df


def merge_back_to_back_comments(df: pd.DataFrame) -> pd.DataFrame:
    def extract_group_id(comment_id):
        return comment_id.split("_")[0] if "_" in comment_id else comment_id

    # Extract group ID from the comment ID
    df["message_id"] = df["id"].apply(extract_group_id)

    # Group by the group ID and aggregate selected columns
    merged = df.groupby("message_id", as_index=False).agg(
        {
            "conv_id": "first",
            "speaker": "first",
            "role": "first",
            "text": " ".join,
            "dialogue act": ",".join,
            # change to 1 if any is 1, leave -1 if all are -1
            "informational motive": "max",
            "social motive": "max",
            "coordinative motive": "max",
        }
    )
    merged = merged.rename(
        columns={
            "informational motive": "informational_motive",
            "social motive": "social_motive",
            "coordinative motive": "coordinative_motive",
            "dialogue act": "dialogue_act",
        }
    )

    return merged


def expand_dialogue_acts(df: pd.DataFrame) -> pd.DataFrame:
    # for each row where is_moderator is True
    # for each value in the dialogue_act string
    # select unique values
    # match each value (which is an index on the DIALOGUE_PATH array)
    # then create columns for each of the values and set it to 1 where
    #  the index is present

    # Ensure dialogue_act is a string and handle missing values
    df.dialogue_act = df.dialogue_act.fillna("").astype(str)

    # Initialize new columns for all dialogue acts with False
    for act in DIALOGUE_ACTS:
        df[act] = False

    # Only expand for moderator rows
    moderator_rows = df["is_moderator"]

    # Go through each row and set True where the dialogue act index appears
    for idx, row in df.loc[moderator_rows].iterrows():
        acts_str = row["dialogue_act"]
        if acts_str.strip() == "":
            continue

        # Split by comma, strip spaces, and filter out empties
        act_indices = [
            a.strip() for a in acts_str.split(",") if a.strip().isdigit()
        ]

        # Set the corresponding columns to True if index exists
        for a in act_indices:
            act_idx = int(a)
            if 0 <= act_idx < len(DIALOGUE_ACTS):
                df.at[idx, DIALOGUE_ACTS[act_idx]] = True

    return df


def main():
    df = import_df(INPUT_DIR)
    df = df.astype(str)

    df = merge_back_to_back_comments(df)

    df["is_moderator"] = df.role == "mod"
    df["moderation_supported"] = True

    df["user"] = df.speaker.apply(util.preprocessing.hash_to_md5)

    df["speaker_turn"] = df.groupby("conv_id").cumcount() + 1
    df["message_id"] = df.apply(
        lambda row: f"whow-{row.get('conv_id')}-{row.get('speaker_turn')}",
        axis=1,
    )
    df = df.sort_values(by="message_id")
    df["reply_to"] = util.preprocessing.assign_reply_to(
        df,
        conv_id_col="conv_id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )

    df.informational_motive = df.informational_motive.apply(lambda x: x == 1)
    df.social_motive = df.social_motive.apply(lambda x: x == 1)
    df.coordinative_motive = df.coordinative_motive.apply(lambda x: x == 1)
    df = expand_dialogue_acts(df)

    df["notes"] = util.preprocessing.notes_from_columns(
        df,
        [
            "informational_motive",
            "social_motive",
            "coordinative_motive",
        ] + DIALOGUE_ACTS,
    )

    df["dataset"] = "whow"
    df["escalated"] = False
    df["escalation_supported"] = False

    df = util.preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
