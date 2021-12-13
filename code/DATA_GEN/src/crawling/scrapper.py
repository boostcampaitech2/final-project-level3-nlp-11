import os
import re
import time
import warnings
import pandas as pd
import json
import argparse

from tqdm import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class Google_crawling:
    def __init__(self, args) -> None:
        self.main(args)

    def start_search(self, driver, search_text):
        URL = "https://www.google.com/maps/"
        driver.get(URL)

        search = driver.find_element_by_css_selector(
            "input#searchboxinput.tactile-searchbox-input"
        )
        time.sleep(1)
        search.clear()
        search.send_keys(search_text)
        search.send_keys(Keys.ENTER)

        driver.implicitly_wait(3)

        return self.get_review_data(driver, search_text)

    def get_review_data(self, driver, search_text):
        while True:
            try:
                time.sleep(5)
                more_review_btn = driver.find_element_by_css_selector(
                    "button[aria-label*='리뷰 더보기']"
                )
                more_review_btn.send_keys(Keys.ENTER)
            except Exception as e:
                print(e)
                break
        last_height = -1
        cnt = 0
        for i in range(100):
            try:
                scroll = driver.find_element_by_css_selector(
                    "div.siAUzd-neVct.section-scrollbox.cYB2Ge-oHo7ed.cYB2Ge-ti6hGc"
                )

                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", scroll
                )
                new_height = driver.execute_script(
                    "return arguments[0].scrollHeight", scroll
                )
                if new_height == last_height:
                    cnt += 1
                    if cnt >= 10:
                        break
                else:
                    last_height = new_height
                    cnt = 0
            except Exception as e:
                print(e)
                break
            time.sleep(1)

        reviews = driver.find_elements_by_css_selector("span.ODSEW-ShBeI-text")
        stars = driver.find_elements_by_css_selector("span.ODSEW-ShBeI-H1e3jb")
        data = [
            [
                search_text,
                re.search(r"[0-9]", star.get_attribute("aria-label")).group(),
                review.text.replace("\n", " "),
            ]
            for star, review in zip(stars, reviews)
        ]
        return data

    def main(self, args):
        data_path = os.getenv("DATA_GEN_DATA_PATH")
        file_path = data_path + args.name_path
        if args.num == -1:
            print("개인 번호를 입력해주세요.")
            return
        with open(file_path, "r", encoding="utf-8") as f:
            spots = json.load(f)
            search_texts = []
            loc = "서울"
            all_cnt = 0
            if args.num == 0:
                locs = list(spots.keys())[:8]
            elif args.num == 1:
                locs = list(spots.keys())[8:9]
            elif args.num == 2:
                locs = list(spots.keys())[9:10]
            elif args.num == 3:
                locs = list(spots.keys())[10:12]
            elif args.num == 4:
                locs = list(spots.keys())[12:13]
            elif args.num == 5:
                locs = list(spots.keys())[13:15]
            elif args.num == 6:
                locs = list(spots.keys())[15:]
            elif args.num == 7:
                locs = list(spots.keys())
            for loc in locs:
                cnt = 0
                for theme in spots[loc].keys():
                    cnt += len(spots[loc][theme])
                all_cnt += cnt
            print(f"{args.num} {all_cnt}")
            for loc in locs:
                for theme in spots[loc].keys():
                    spot = spots[loc][theme]
                    for s in spot:
                        if s != "totalCount":
                            text = f"{loc} {s}"
                            search_texts.append([text, s, loc, theme])

        # chrome_options setting for server
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # 자신의 크롬드라이브 위치

        code_path = os.getenv("DATA_GEN_CODE_PATH")
        path = code_path + args.driver_path
        driver_path = path
        driver = webdriver.Chrome(driver_path, chrome_options=chrome_options)
        for search_text, spot, loc, theme in tqdm(search_texts):
            spot = spot.replace("/", " ")
            theme = theme.replace("/", "_")
            if not os.path.exists(f"{data_path}{loc}"):
                os.makedirs(f"{data_path}{loc}")
            if not os.path.exists(f"{data_path}{loc}/{theme}"):
                os.makedirs(f"{data_path}{loc}/{theme}")
            if not f"{spot}.csv" in os.listdir(f"{data_path}{loc}/{theme}"):
                data = self.start_search(driver, search_text)
                df = pd.DataFrame(data, columns=["place", "star", "review"])
                df.to_csv(f"{data_path}{loc}/{theme}/{spot}.csv")
            else:
                print(f"{spot}.csv is already exists!!")


if __name__ == "__main__":
    Google_crawling()
