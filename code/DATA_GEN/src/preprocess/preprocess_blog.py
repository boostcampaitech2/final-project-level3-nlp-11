import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from src.crawling.utils.replace_text import ReplaceText

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class PreprocessBlog:
    def __init__(self, args) -> None:
        data, info = self.get_data(args)
        self.preprocess(data, info, args)

    def get_data(self, args):
        print("------------------READ DATA FILE ------------------")
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(file_path + args.input_result, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        with open(file_path + args.input_info, "r", encoding="utf-8-sig") as f:
            info = json.load(f)
        print("------------------FIN READ DATA FILE ------------------")
        return data, info

    def preprocess(self, data, info, args):
        result = {}
        result_info = {}
        for state in tqdm(info.keys()):
            if not state in result:
                result[state] = {}
                result_info[state] = {}
            for types in info[state].keys():
                if not types in result[state]:
                    result[state][types] = {}
                    result_info[state][types] = {}
                for location in info[state][types].keys():
                    if info[state][types][location]["total"] < 100:
                        continue
                    if not location in result[state][types]:
                        result[state][types][location] = []
                        result_info[state][types][location] = {}

                    for i in data[state][types][location]:
                        i["context"] = ReplaceText().get_convert_text(i["context"])
                        result[state][types][location].append(i)
                    result_info[state][types][location] = info[state][types][location]

        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(file_path + args.output_result, "w", encoding="utf-8-sig") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        with open(file_path + args.output_info, "w", encoding="utf-8-sig") as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    PreprocessBlog()
