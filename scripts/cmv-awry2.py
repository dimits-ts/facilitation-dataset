from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import py3langid as langid


def is_english(text: str) -> bool:
    try:
        lang, prob = langid.classify(text)
        return lang == 'en' and prob > 0.8
    except OverflowError:
        return False

def read_df(input_dir: Path) -> pd.DataFrame:
    df_ls = []
    for file in tqdm(list(INPUT_DIR.rglob("*.json"))):
        df_ls.append(pd.read_json(file, encoding="utf-8", lines=True))
    return pd.concat(df_ls)


def get_user_id(user_dict: dict[str, str]) -> str:
    if user_dict is None:
        return None
    if type(user_dict) == type(float):
        return user_dict
    
    if "id" in user_dict:
        return user_dict["id"]
    elif "ip" in user_dict:
        return user_dict["ip"]
    else:
        return None


INPUT_DIR = Path("../downloads/cmv_awry2")
OUTPUT_PATH = Path("../datasets/cmv_awry2.csv")


def main():
    print("Reading dataset...")
    df = read_df(INPUT_DIR)

    print("Processing dataset...")
    df.user = df.user.apply(get_user_id)

    df = df.loc[:, ["id", "replytoId", "comment", "user", "score"]]
    df = df.rename(columns={"conv_id", "replyTo", "text", "user", "notes"})
    # do not attempt multiprocessing unless you have ungodly amounts of RAM
    df = df[df.content.progress_apply(lambda text: is_english(text))]

    print("Exporting...")
    df.to_csv(OUTPUT_PATH)

    print("Done.")


if __name__ == "__main__":
    main()