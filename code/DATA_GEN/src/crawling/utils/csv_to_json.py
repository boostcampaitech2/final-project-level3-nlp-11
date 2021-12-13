import os
import json
import pandas as pd
import unicodedata
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class CsvToJson:
    def __init__(self, args) -> None:
        df = self.get_data(args)
        self.find_target(df, args)

    def get_data(self, args):
        print("------------------READ DATA FILE ------------------")
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        df = pd.read_csv(file_path + args.csv_file_name)
        print("------------------FIN READ DATA FILE ------------------")
        return df

    def find_target(self, df, args):
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
        with open(
            file_path + args.output_json_file_name, "w", encoding="utf-8-sig"
        ) as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return
