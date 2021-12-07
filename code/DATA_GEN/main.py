import argparse
import configparser

from src.crawling.blog_crawling import Crawling_naver
from src.crawling.cafe_crawling import CafeCrawling
from src.crawling.scrapper import Google_crawling


def main(args):
    # 모든 데이터 크롤링 코드 (엄청 오래걸림)
    crawling(args)

    # blog 데이터 전처리
    # google 데이터 전처리
    # cafe 데이터 전처리

    return


def crawling(args):
    Crawling_naver(args)
    CafeCrawling(args)
    Google_crawling(args)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="./src/config.conf", help="config file"
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
        "--name_path", type=str, default="tour_spot_name.json", help="set start_point"
    )
    parser.add_argument(
        "--info_path", type=str, default="test_info.json", help="set start_point"
    )
    parser.add_argument(
        "--result_path", type=str, default="test_result.json", help="set start_point"
    )
    parser.add_argument(
        "--driver_path",
        type=str,
        default="src/crawling/utils/chromedriver",
        help="chromedriver_path",
    )
    parser.add_argument(
        "--num", default=-1, type=int, help="An number for seperating task"
    )

    for k, v in config.items("crawling"):
        parser.parse_args([str(k), str(v)], args)

    args = parser.parse_args(left_argv, args)
    main(args)
