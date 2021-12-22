import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from src.crawling.utils.replace_text import ReplaceText

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class PreprocessGoogle:
    def __init__(self, args) -> None:
        data = self.get_data(args)
        self.preprocess(data, args)

    def get_data(self, args):
        print("------------------READ DATA FILE ------------------")
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(
            file_path + args.output_json_file_name, "r", encoding="utf-8-sig"
        ) as f:
            data = json.load(f)
        print("------------------FIN READ DATA FILE ------------------")
        return data

    def preprocess(self, data, args):
        result = {}
        for state in tqdm(data.keys()):
            if not state in result:
                result[state] = {}
            for types in data[state].keys():
                if not types in result[state]:
                    result[state][types] = {}
                for location in data[state][types].keys():
                    if not location in result[state][types]:
                        result[state][types][location] = []
                    for i in data[state][types][location]:
                        if i["rating"] < args.minimun_google_rating:
                            continue
                        context = ReplaceText().get_convert_text(i["review"])
                        if not context:
                            continue
                        i["review"] = context

                        result[state][types][location].append(i)

        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(file_path + args.output_google, "w", encoding="utf-8-sig") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    PreprocessGoogle()
