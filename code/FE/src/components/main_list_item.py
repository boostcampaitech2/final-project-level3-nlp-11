import os
import urllib.parse
import streamlit as st

from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def main_list_item(data, area, user_query=None):
    root_path = os.getenv("ROOT_PATH")

    spot = data["spot"]
    location = data["location"]
    context = data["context"]
    addr = data["addr1"]
    tel = data["tel"]
    context = context.replace("\n", "<br>")
    img_url = data["img_url"]
    style_img = ""
    if img_url == "https://i.imgur.com/wbN96ze.png":
        style_img = "opacity:0.5;border:1px solid rgb(29,51,63,.4);padding-top:45px;padding-bottom:45px"
    query = urllib.parse.quote_plus(addr + " " + spot)
    if tel == "전화 정보 없음":
        tmp = f"<b>Tel</b>: {tel}"
    else:
        tmp = f"""  <a href="tel:{tel}">
                        <b>Tel</b>: {tel}
                    </a>"""
    return f"""
            <div style="clear:both;padding:10px;padding-top:20px">
                <div id=context_img >
                    <a href="{root_path}?page=2&target={spot}&location={area}&query={user_query}" target="_blank">
                        <img src="{img_url}" width="100%" style="border-radius:10px;vertical-align:middle;{style_img}" >
                    </a>
                </div>
                <div id=context >
                    <a href="{root_path}?page=2&target={spot}&location={area}&query={user_query}"target="_blank"style="text-decoration:none;color:rgb(29,51,63)">
                        <p style="font-size:38px;font-weight:bold;">
                            {spot}
                        </p>
                        <p id="1" style="font-size:15px;display:-webkit-box;overflow:hidden;white-space:normal;-webkit-line-clamp:2;-webkit-box-orient:vertical;"> 
                            {context}
                        </p>
                    </a>
                    <p>
                        <a href="https://map.naver.com/v5/search/{query}?" style="text-decoration:none;color:rgb(29,51,63)" target="_blank">
                            <b>Address</b> : {addr} <br>
                        </a>
                        {tmp}
                    </p>
                    <a href="https://map.naver.com/v5/search/{query}?" target="_blank">
                        <img src="https://t1.daumcdn.net/cfile/tistory/99E493445C0D20A143" style="width:35px">
                    </a>
                </div>
            </div>"""


def parser_data(data, area, user_query=None):
    if not data:
        return ""
    html = ""
    for i in data:
        html += main_list_item(i, area, user_query)

    return html
