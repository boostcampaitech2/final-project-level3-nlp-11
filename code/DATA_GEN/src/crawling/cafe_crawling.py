import json
import os
import re
import time
import pandas as pd
import numpy as np
import warnings

from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime
from contextlib import suppress
from dotenv import load_dotenv
from src.crawling.utils.replace_text import ReplaceText

warnings.filterwarnings("ignore")
env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


class CafeCrawling:
    def __init__(self, args) -> None:
        data_path = os.getenv("DATA_GEN_DATA_PATH")
        file_path = data_path + args.name_path
        driver_path = args.driver_path
        with open(file_path, "rt", encoding="UTF8") as json_file:
            json_data = json.load(json_file)
            self.run_crawling(json_data, driver_path)

    def run_crawling(self, json_data, driver_path):

        code_path = os.getenv("DATA_GEN_CODE_PATH")
        path = code_path + driver_path
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--single-process")
        chrome_options.add_argument("--disable-dev-shm-usage")
        browser = webdriver.Chrome(path, chrome_options=chrome_options)

        seoul_tour = json_data["서울"]["관광지"]
        seoul_festival = json_data["서울"]["행사/공연/축제"]
        seoul_reports = json_data["서울"]["레포츠"]

        seoul_tour = self.data_make(seoul_tour)
        seoul_festival = self.data_make(seoul_festival)
        seoul_reports = self.data_make(seoul_reports)

        seoul_tour_final = {}
        for s in seoul_tour:
            seoul_tour_texts = []
            for page in tqdm(range(1, 11)):
                url = self.move_page(s, page)
                browser.get(url)
                time.sleep(0.3)
                article_list = browser.find_elements_by_css_selector(
                    "a.api_txt_lines.total_tit"
                )
                for article in article_list:
                    article.click()
                    time.sleep(0.3)
                    change_tab = browser.window_handles[-1]
                    browser.switch_to.window(change_tab)

                    try:
                        data = self.get_data(browser)
                        seoul_tour_texts.append(data)
                    except:
                        pass
                    browser.close()
                    change_tab = browser.window_handles[-1]
                    browser.switch_to.window(change_tab)

            seoul_tour_final[s] = seoul_tour_texts
        temp = seoul_tour_final

        temp_result = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in temp.items()]))
        temp_result

        final = temp_result.T

        final.to_csv("seoul_tour_naver_cafe.csv", mode="w")

    def data_make(self, seoul):
        seoul_data = []
        for i, s in enumerate(seoul):
            if i == 0:
                continue
            else:
                seoul_data.append(s)
        return seoul_data

    def move_page(self, searching, page):
        url = f"https://search.naver.com/search.naver?where=article&ie=utf8&query={searching}&prdtype=0&t=0&st=rel&date_option=0&date_from=&date_to=&srchby=text&dup_remove=1&cafe_url=&without_cafe_url=&board=&sm=tab_pge&start={(page*10)-9}"
        return url

    def get_data(self, browser):
        browser.switch_to.frame("cafe_main")
        r = browser.page_source
        page_soup = BeautifulSoup(r, "html.parser")
        content = page_soup.find("div", class_="ArticleContentBox")
        content = content.find("div", class_="article_viewer").text.strip()
        final = ReplaceText().get_convert_text(content)
        return final
