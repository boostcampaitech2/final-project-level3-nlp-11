import argparse
import configparser

from src.crawling.blog_crawling import Crawling_naver
from src.crawling.cafe_crawling import CafeCrawling
from src.crawling.scrapper import Google_crawling
from src.preprocess.preprocess_blog import PreprocessBlog
from src.crawling.utils.csv_to_json import CsvToJson


def main(args):
    # 모든 데이터 크롤링 코드 (엄청 오래걸림)
    if args.run_crawling:
        crawling(args)

    # blog,google,caf 데이터 전처리
    if args.run_preprocess:
        preprocess(args)

    # google to json
    CsvToJson(args)

    if args.run_mk_pair:
        return
        # make pair
    return


def crawling(args):
    Crawling_naver(args)
    CafeCrawling(args)
    Google_crawling(args)
    return


def preprocess(args):

    PreprocessBlog(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="./config/config.conf", help="config file"
    )
    args, left_argv = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, "r") as f:
            config = configparser.SafeConfigParser()
            config.read([args.config_file])

    parser.add_argument("--start_state", type=str, default="서울", help="set start_point")
    parser.add_argument("--start_type", type=str, default="관광지", help="set start_point")
    parser.add_argument(
        "--start_location", type=str, default="간데메공원", help="set start_point"
    )
    parser.add_argument(
        "--use_start_point",
        type=bool,
        default=False,
        help="use start_point",
    )
    parser.add_argument(
        "--name_path",
        type=str,
        default="tour_spot_name.json",
        help="original name file name",
    )
    parser.add_argument(
        "--info_path", type=str, default="test_info.json", help="new info file name"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="test_result.json",
        help="new result file name",
    )
    parser.add_argument(
        "--driver_path",
        type=str,
        default="src/crawling/utils/chromedriver",
        help="chromedriver_path",
    )
    parser.add_argument(
        "--num", default=7, type=int, help="An number for seperating task"
    )

    for k, v in config.items("crawling"):
        parser.parse_args([str(k), str(v)], args)

    parser.add_argument(
        "--input_result",
        type=str,
        default="result_latest.json",
        help="input_result",
    )
    parser.add_argument(
        "--input_info",
        type=str,
        default="info_latest.json",
        help="input_info",
    )
    parser.add_argument(
        "--output_result",
        type=str,
        default="result_prepro.json",
        help="output_result",
    )
    parser.add_argument(
        "--output_info",
        type=str,
        default="info_prepro.json",
        help="output_info",
    )

    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="review.csv",
        help="csv_file_name",
    )

    parser.add_argument(
        "--output_json_file_name",
        type=str,
        default="review.json",
        help="output_json_file_name",
    )

    for k, v in config.items("preprocess"):
        parser.parse_args([str(k), str(v)], args)

    parser.add_argument(
        "--run_crawling",
        type=bool,
        default=False,
        help="To make original file default False",
    )

    parser.add_argument(
        "--run_preprocess",
        type=bool,
        default=False,
        help="run preprocess in src/preprocess default False",
    )

    parser.add_argument(
        "--run_mk_pair",
        type=bool,
        default=True,
        help="run mk_pair in src/mk_pair default True",
    )
    for k, v in config.items("progress"):
        parser.parse_args([str(k), str(v)], args)

    args = parser.parse_args(left_argv, args)
    main(args)
