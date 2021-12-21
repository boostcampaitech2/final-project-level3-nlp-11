import json
import os
from tqdm import tqdm
from kss import split_chunks
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class SplitChunck:
    def __init__(self) -> None:
        dataset, result = self.get_data()
        self.split(dataset, result)

    def get_data(self):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        print("------------------READ DATA FILE ------------------")
        with open(file_path + "12_16/pair.json", "r", encoding="utf-8-sig") as f:
            dataset = json.load(f)

        with open(file_path + "test_result.json", "r", encoding="utf-8-sig") as f:
            result = json.load(f)

        print("------------------FIN READ DATA FILE ------------------")
        return dataset, result

    def split(self, dataset, result):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        flag = True
        for state in dataset.keys():
            if state != "부산" and flag:
                continue
            flag = False
            if not state in result:
                result[state] = {}

            for types in dataset[state].keys():
                if not types in result[state]:
                    result[state][types] = {}

                for location in tqdm(
                    dataset[state][types].keys(), desc=f"{state}-{types}"
                ):
                    if not location in result[state][types]:
                        result[state][types][location] = []

                    tmp = []
                    for context in dataset[state][types][location]:
                        try:
                            split = split_chunks(
                                context["context"], backend="mecab", max_length=1024
                            )
                        except:
                            split = [
                                "kss 오류로 인한 split chunk 불가 문장입니다.",
                                context["context"],
                            ]
                        tmp.append(
                            {
                                "query": context["query"],
                                "context": split,
                                "url": context["url"],
                            }
                        )
                    result[state][types][location].extend(tmp)

                with open(
                    file_path + "test_result.json", "w", encoding="utf-8-sig"
                ) as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    SplitChunck().split()
