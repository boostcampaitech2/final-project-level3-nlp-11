import os
import urllib.parse
import streamlit as st

from styles.main_list import main_list_style
from components.main_list_item import parser_data
from api.index import Api
from model.model_index import get_data, get_model
from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def info_page(query_params):
    with st.spinner("찾는 중!..."):
        spot = query_params["target"][0]
        area = query_params["location"][0]
        result_contents = ""
        result = Api().get_tour_topK([spot], area)
        data_path = os.getenv("DATA_PATH")
        retriever = get_model(data_path)
        result_contents = get_data(retriever, spot, area)
        img_url = result[0]["img_url"]
        context = result[0]["context"]
        addr = result[0]["addr1"]
        tel = result[0]["tel"]
        context = context.replace("\n", "<br>")
        query = urllib.parse.quote_plus(addr + " " + spot)

        if tel == "전화 정보 없음":
            tmp = f"<b>Tel</b>: {tel}"
        else:
            tmp = f"""  <a href="tel:{tel}">
                            <b>Tel</b>: {tel}
                        </a>"""

    link = f"""
            <div style="font-size:38px;font-weight:bold;clear:both;padding-top:50px;padding-left:20px"> {spot} </div>
            {main_list_style()}
            
            <div style="font-size:62.5%">
                <div style="height:100%;clear:both;padding:10px">
                    <div id=context_img>
                        <img src="{img_url}" width="100%" style="border-radius:10px;vertical-align:middle" >
                    </div>
                    <div id=context>
                        <p id="1" style="font-size:15px"> 
                            {context}
                        </p>
                        <p>
                            <a href="https://map.naver.com/v5/search/{query}?" style="text-decoration:none;color:rgb(29,51,63)" target="_blank">
                                <b>Address</b> : {addr} <br>
                            </a>
                            {tmp}
                        </p>
                        <a href="https://map.naver.com/v5/search/{query} "target="_blank">
                            <img src="https://t1.daumcdn.net/cfile/tistory/99E493445C0D20A143" style="width:35px">
                        </a>
                    </div>
                </div>
            </div>         
            <hr style="clear:both;padding-top:30px">
            <div style="font-size:38px;font-weight:bold;clear:both;padding-top:10px;padding-left:20px"> {area} 관련 추천 명소 </div>
            
            <div style="font-size:62.5%;">
                {parser_data(result_contents,area)}
            </div>

            """

    st.markdown(link, unsafe_allow_html=True)
