from pathlib import Path

import util.io


INPUT_PATH = Path("../pefk.csv")


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def main():
    print("Loading dataset to extract statistics...")
    df = util.io.progress_load_csv(INPUT_PATH)

    print("*" * 25)
    print("Comments per discussion:")
    print(df.groupby("conv_id").size().describe())

    print("*" * 25)
    print("Comments per dataset:")
    print(df.groupby("dataset")["message_id"].nunique())

    print("*" * 25)
    print("Discussions per dataset:")
    print(df.groupby("dataset")["conv_id"].nunique())

    print("*" * 25)
    print("Word count per comment:")
    print(
        df.text
        .astype(str)
        .apply(lambda x: x.split())
        .apply(len)
        .astype(int)
        .describe()
    )

    print("*" * 25)
    print("Percentage of moderator comments per dataset:")
    moderator_percent = df.groupby("dataset")["is_moderator"].mean() * 100
    print(moderator_percent.round(2).astype(str) + " %")

    print("*" * 25)
    print(f"Dataset total size: {convert_bytes(INPUT_PATH.stat().st_size)}")


if __name__ == "__main__":
    main()
