from pathlib import Path

import pandas as pd

import util.preprocessing


INPUT_PATH = Path("../downloads/wikitactics/wikitactics.json")
OUTPUT_PATH = Path("../datasets/wikitactics.csv")


def is_moderator(
    rebuttal_labels: list[str], coordination_labels: list[str]
) -> bool:
    MODERATION_COORDINATION = [
        "Asking questions",
        "Coordinating edits",
        "Providing clarification",
        "Suggesting a compromise",
        "Contextualisation",
    ]
    MODERATION_REBUTTAL = [
        "DH6: Refutation of opponent's argument (with evidence or reasoning)",
        "DH5: Counterargument with new evidence / reasoning",
        "DH7: Refuting the central point",
    ]

    # Check if any label in rebuttal_labels is in MODERATION_REBUTTAL
    if rebuttal_labels is not None and any(
        label in MODERATION_REBUTTAL for label in rebuttal_labels
    ):
        return True

    # Check if any label in coordination_labels is in MODERATION_COORDINATION
    if coordination_labels is not None and any(
        label in MODERATION_COORDINATION for label in coordination_labels
    ):
        return True

    return False


def main():
    df = pd.read_json(INPUT_PATH)
    df = df.explode(column="utterances")

    utterance_df = pd.json_normalize(df.utterances)
    df = pd.concat([df.reset_index(), utterance_df.reset_index()], axis=1)
    df = df.drop(columns=["utterances", "index"])

    df["is_moderator"] = df.apply(
        lambda row: is_moderator(
            row.get("rebuttal_labels"), row.get("coordination_labels")
        ),
        axis=1,
    )
    df["moderation_supported"] = True

    df["speaker_turn"] = df.groupby("conv_id").cumcount() + 1
    # make sure message_id is unique across discussions
    df["message_id"] = df.apply(
        lambda row: f"wikitactics-{row.get('conv_id')}-"
        f"{row.get('speaker_turn')}",
        axis=1,
    )
    df["reply_to"] = util.preprocessing.assign_reply_to(
        df,
        conv_id_col="conv_id",
        message_id_col="message_id",
        order_col="speaker_turn",
    )
    df["dataset"] = "wikitactics"
    df["notes"] = None
    df["escalated"] = df["escalation_label"]
    df["escalation_supported"] = True
    df = df.rename(columns={"username": "user"})
    df = util.preprocessing.std_format_df(df)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
