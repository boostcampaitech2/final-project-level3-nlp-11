import json
import os
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class DAPT:
    def __init__(self) -> None:
        review, result = self.get_data()
        self.json_to_text(review, result)

    def get_data(self):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        print("------------------READ DATA FILE ------------------")
        with open(
            file_path + "original/google_prepro.json", "r", encoding="utf-8-sig"
        ) as f:
            review = json.load(f)

        with open(
            file_path + "original/result_prepro.json", "r", encoding="utf-8-sig"
        ) as f:
            result = json.load(f)

        print("------------------FIN READ DATA FILE ------------------")
        return review, result

    def json_to_text(self, review, result):
        text_str = ""
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        for state in review.keys():
            for types in review[state].keys():
                for location in review[state][types].keys():
                    for i, v in enumerate(review[state][types][location]):
                        text_str += v["review"] + "\n"

        for state in result.keys():
            for types in result[state].keys():
                for location in result[state][types].keys():
                    for i, v in enumerate(result[state][types][location]):
                        text_str += v["context"] + "\n"

        with open(file_path + "DAPT_DOCS.txt", "w") as f:
            f.write(text_str)


if __name__ == "__main__":
    DAPT()
