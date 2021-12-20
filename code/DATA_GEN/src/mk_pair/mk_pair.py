import json
import os
import numpy as np
from tqdm import tqdm
import re
from numpy.lib.function_base import append

from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Mecab
from dotenv import load_dotenv

env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class MkPair:
    def __init__(self, args) -> None:
        dataset, info, review = self.get_data(args)
        self.get_result(dataset, info, review, args)

    def get_data(self, args):
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        print("------------------READ DATA FILE ------------------")
        with open(file_path + args.output_result, "r", encoding="utf-8-sig") as f:
            dataset = json.load(f)

        with open(file_path + args.output_google, "r", encoding="utf-8-sig") as f:
            review = json.load(f)

        with open(file_path + args.output_info, "r", encoding="utf-8-sig") as f:
            info = json.load(f)
        print("------------------FIN READ DATA FILE ------------------")
        return dataset, info, review

    def sparse(self, dataset, review, args):
        mecab = Mecab()
        tokenizer_func = lambda x: mecab.morphs(x)
        corpus = []
        raw_corpus = []
        queries = []
        raw_queries = []
        url = []
        for pair in dataset:

            if args.clean_tokenizer:
                context = pair["context"]
                context = re.sub(r"""([^a-zA-Z 0-9ㄱ-ㅎ|ㅏ-ㅣ|가-힣])""", "", context)
                raw_corpus.append(pair["context"])
                corpus.append(context)
            else:
                raw_corpus.append(pair["context"])
                corpus.append(pair["context"])

            url.append(pair["url"])
        for pair in review:
            context = re.findall(r"""[ ]""", review)
            if len(context) < 5:
                continue

            if args.clean_tokenizer:
                review = pair["review"]
                review = re.sub(r"""([^a-zA-Z 0-9ㄱ-ㅎ|ㅏ-ㅣ|가-힣])""", "", review)
                queries.append(review)
                raw_queries.append(pair["review"])
            else:
                queries.append(pair["review"])
                raw_queries.append(pair["review"])

        if len(queries) < args.minimun_pair_num:
            return None
        vectorizer = TfidfVectorizer(tokenizer=tokenizer_func, ngram_range=(1, 2))
        vectorizer.fit(corpus)
        sp_matrix = vectorizer.transform(corpus)
        query_vec = vectorizer.transform(queries)
        result = query_vec * sp_matrix.T
        rank_list = np.array(np.argsort(-result.todense(), axis=None))[0]

        pair = []
        duplication_query = []
        duplication_context = []
        for i in rank_list:
            if len(pair) == args.minimun_pair_num:
                break

            x = i // result.shape[1]
            y = i % result.shape[1]

            if x in duplication_query:
                continue
            if y in duplication_context:
                continue
            duplication_context.append(y)
            duplication_query.append(x)
            tmp = {"query": raw_queries[x], "context": raw_corpus[y], "url": url[y]}
            pair.append(tmp)
        if len(pair) < args.minimun_pair_num:
            return None
        return pair

    def get_result(self, dataset, info, review, args):
        result = {}
        result_info = {}
        file_path = os.getenv("DATA_GEN_DATA_PATH")
        for state in info.keys():
            if not state in result:
                result[state] = {}
                result_info[state] = {}

            for types in info[state].keys():
                if types == "문화시설":
                    continue
                if not types in result[state]:
                    result[state][types] = {}
                    result_info[state][types] = {}

                for location in tqdm(
                    info[state][types].keys(), desc=f"{state}-{types}"
                ):
                    if not location in review[state][types].keys():
                        continue

                    if (
                        len(dataset[state][types][location]) < args.minimun_pair_num
                        or len(review[state][types][location]) < args.minimun_pair_num
                    ):
                        continue

                    tmp = self.sparse(
                        dataset[state][types][location],
                        review[state][types][location],
                        args,
                    )
                    if not tmp:
                        continue

                    if not location in result[state][types]:
                        result[state][types][location] = []

                    result[state][types][location] = tmp

                result_info[state][types] = {"len": len(result[state][types])}

            with open(file_path + args.output_pair, "w", encoding="utf-8-sig") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            with open(
                file_path + args.output_pair_info, "w", encoding="utf-8-sig"
            ) as f:
                json.dump(result_info, f, indent=4, ensure_ascii=False)
