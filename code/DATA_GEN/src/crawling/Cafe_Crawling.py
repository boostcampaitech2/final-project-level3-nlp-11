#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json

file_path = r"C:\Users\harry\Desktop\Naver\tour_spot_name_without_cultural.json"
with open(file_path, "rt", encoding="UTF8") as json_file:
    json_data = json.load(json_file)
    print(json_data)


# In[2]:


seoul_tour = json_data["서울"]["관광지"]
seoul_festival = json_data["서울"]["행사/공연/축제"]
seoul_reports = json_data["서울"]["레포츠"]


# In[3]:


# In[4]:


def data_make(seoul):
    seoul_data = []
    for i, s in enumerate(seoul):
        if i == 0:
            continue
        else:
            seoul_data.append(s)
    return seoul_data


# In[5]:


seoul_tour = data_make(seoul_tour)
seoul_festival = data_make(seoul_festival)
seoul_reports = data_make(seoul_reports)


# In[6]:


# In[7]:


def move_page(searching, page):
    url = f"https://search.naver.com/search.naver?where=article&ie=utf8&query={searching}&prdtype=0&t=0&st=rel&date_option=0&date_from=&date_to=&srchby=text&dup_remove=1&cafe_url=&without_cafe_url=&board=&sm=tab_pge&start={(page*10)-9}"
    return url


# In[8]:


from selenium import webdriver
import pyautogui
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
from datetime import datetime
import random

path = r"C:\Users\harry\Downloads\chromedriver_win32\chromedriver.exe"
browser = webdriver.Chrome(path)


# In[9]:


import re


def get_convert_text(context):
    context = context.replace("\n", "")
    context = context.replace("&nbsp;", "")
    # 해쉬태그 제거(한글만)
    context = re.sub(r"""#[ㄱ-ㅎ|ㅏ-ㅣ|가-힣 ]*""", "", context)
    # 공백 2개 한개로 변경
    context = re.sub(r""" +(?= )""", "", context)
    # ULR
    context = re.sub(
        r"""https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)""",
        "",
        context,
    )
    # dict 형식 제거
    context = re.sub(r"""{[^}]*}""", " ", context)
    # 온점 후 띄어쓰기 강제
    # context = re.sub(r"""\.(?=[^ ])""", ". ", context)
    # 아래 나온 문자외 전부 제거
    context = re.sub(r"""[^0-9a-zA-Zㄱ-ㅎ|ㅏ-ㅣ|가-힣 ().,?!]""", "", context)
    return context


# In[10]:


import pickle
from contextlib import suppress


def get_data(browser):
    browser.switch_to.frame("cafe_main")
    html = browser.page_source
    # print(html)
    soup = BeautifulSoup(html, "html.parser")
    r = browser.page_source
    page_soup = BeautifulSoup(r, "html.parser")
    content = page_soup.find("div", class_="ArticleContentBox")
    # print(content)
    contents_list = []
    temp_dict = {}
    # temp_dict['content'] = content.find("div", class_="article_viewer").text.strip()
    content = content.find("div", class_="article_viewer").text.strip()
    final = get_convert_text(content)
    # print(final)
    # contents_list.append(temp_dict)
    # print(contents_list)
    return final


# In[11]:


seoul_tour_final = {}
for s in seoul_tour:
    seoul_tour_texts = []
    for page in tqdm(range(1, 11)):
        url = move_page(s, page)
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
                data = get_data(browser)
                seoul_tour_texts.append(data)
            except:
                pass
            browser.close()
            change_tab = browser.window_handles[-1]
            browser.switch_to.window(change_tab)
            # print(seoul_tour_texts)
    seoul_tour_final[s] = seoul_tour_texts
    print(seoul_tour_final)


# In[15]:


emergency = seoul_tour_final


# In[16]:


print(len(seoul_tour_final))


# In[17]:


temp = seoul_tour_final


# In[26]:


temp["약수사(서울)"]


# In[18]:


temp_result = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in temp.items()]))


# In[19]:


temp_result


# In[20]:


import numpy as np

final = temp_result.T


# In[21]:


# final


# In[22]:


final.to_csv("seoul_tour_naver_cafe.csv", mode="w")


# In[27]:


# final


# In[28]:


final.to_csv("./result.csv")


# In[23]:


"""
for s in seoul_festival:
    for page in tqdm(range(1,11)):
        url = move_page(s, page)
        browser.get(url)
        time.sleep(2)
        article_list = browser.find_elements_by_css_selector('a.api_txt_lines.total_tit')
        for article in article_list:
            article.click()
            time.sleep(1+random.uniform(1,2))
            change_tab = browser.window_handles[-1]
            browser.switch_to.window(change_tab)

            try:
                data = get_data(browser)
            except:
                pass
            browser.close()
            change_tab = browser.window_handles[-1]
            browser.switch_to.window(change_tab)
"""


# In[ ]:


"""
for s in seoul_reports:
    for page in tqdm(range(1,11)):
        url = move_page(s, page)
        browser.get(url)
        time.sleep(2)
        article_list = browser.find_elements_by_css_selector('a.api_txt_lines.total_tit')
        for article in article_list:
            article.click()
            time.sleep(1+random.uniform(1,2))
            change_tab = browser.window_handles[-1]
            browser.switch_to.window(change_tab)

            try:
                data = get_data(browser)
            except:
                pass
            browser.close()
            change_tab = browser.window_handles[-1]
            browser.switch_to.window(change_tab)
"""
