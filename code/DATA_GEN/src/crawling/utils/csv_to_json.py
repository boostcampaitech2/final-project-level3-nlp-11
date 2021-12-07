import os
import json
import pandas as pd
import unicodedata
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def main():
    data, info, df = get_data()
    find_target(info, data, df)


def get_data():
    print("------------------READ DATA FILE ------------------")
    file_path = os.getenv("DATA_GEN_DATA_PATH")
    with open(file_path + "test_result_prepro.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    with open(file_path + "test_info_prepro.json", "r", encoding="utf-8-sig") as f:
        info = json.load(f)
    df = pd.read_csv(file_path + "all_reviews.csv")
    print("------------------FIN READ DATA FILE ------------------")
    return data, info, df


def find_target(info, data, df):
    result = {}
    for row in df.itertuples():
        location = unicodedata.normalize("NFC", row.location)
        theme = unicodedata.normalize("NFC", row.theme)
        spot = unicodedata.normalize("NFC", row.spot)
        if theme == "행사 공연 축제":
            theme = "행사/공연/축제"
        if not location in result:
            result[location] = {}
        if not theme in result[location]:
            result[location][theme] = {}
        if not spot in result[location][theme]:
            result[location][theme][spot] = []
        doc = {"rating": row.rating, "review": row.review}
        result[location][theme][spot].append(doc)
    file_path = os.getenv("DATA_GEN_DATA_PATH")
    with open(file_path + "google.json", "w", encoding="utf-8-sig") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return


if __name__ == "__main__":
    main()
