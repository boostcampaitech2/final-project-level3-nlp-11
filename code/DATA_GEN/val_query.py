# 우선 오리지널 google prepro 셋 가져오기 #고정
# !, ., ?, ~ 등 기본적인 것 하나로 바꾸는 코드 구성 #고정
# 먼저 찾고 나서 pair셋 이랑 val set 둘다 같은 전처리 진행
# pair 셋 가져오는거 args, 만들기
# 폴더 구분 확실히 해서 편하게 구성
# -> 기존 폴더 1,2,3,4... 번호로 구분
# 내부에 파일은 pair, pair_old, qeury_set
# 두 파일 업로드
import os
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class ValQuery:
    def __init__(self, dir_name) -> None:
        google_prepro = self.get_google_prerpro()
        pair_old = self.get_pair(dir_name)
        self.filter_dup(google_prepro, pair_old, dir_name)

    def get_google_prerpro(self):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(
            file_path + "original/google_prepro.json", "r", encoding="utf-8-sig"
        ) as f:
            google_prepro = json.load(f)

        return google_prepro

    def get_pair(self, dir_name):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        with open(
            file_path + f"{dir_name}/pair_old.json", "r", encoding="utf-8-sig"
        ) as f:
            pair_old = json.load(f)

        return pair_old

    def filter_dup(self, google_prepro, pair_old, dir_name):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        result = {}
        for state in tqdm(pair_old.keys()):
            if state not in result:
                result[state] = {}
            for types in pair_old[state].keys():
                if types not in result[state]:
                    result[state][types] = {}
                for location in pair_old[state][types].keys():
                    if location not in result[state][types]:
                        result[state][types][location] = []
                    pair_loaction = pair_old[state][types][location]
                    google_location = google_prepro[state][types][location]

                    compair = []
                    for i in google_location:
                        if len(compair) == 5:
                            break
                        if i["rating"] < 3:
                            continue
                        for j in pair_loaction:
                            if i["review"] == j["query"]:
                                compair.append(i["review"])
                                break
                    for i in google_location:
                        if len(result[state][types][location]) > 3:
                            break
                        if i["review"] not in compair:
                            result[state][types][location].append(
                                self.replace_all(i["review"])
                            )

                    tmp = []
                    for i in pair_loaction:
                        info = {
                            "query": self.replace_all(i["query"]),
                            "context": self.replace_all(i["context"]),
                        }
                        tmp.append(info)
                    pair_old[state][types][location] = tmp

            with open(
                file_path + f"{dir_name}/pair.json", "w", encoding="utf-8-sig"
            ) as f:
                json.dump(pair_old, f, indent=4, ensure_ascii=False)

            with open(
                file_path + f"{dir_name}/query_set.json", "w", encoding="utf-8-sig"
            ) as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

    def replace_all(self, context):
        pattern = {
            r"""\.\.""": ".",
            r"""!!""": "!",
            r"""\?\?""": "?",
            r"""~~""": "~",
        }
        for i in pattern.keys():
            context = re.sub(i, pattern[i], context)

        return context


if __name__ == "__main__":
    array = [3, 5, 6]
    for i in array:
        ValQuery(str(i))
