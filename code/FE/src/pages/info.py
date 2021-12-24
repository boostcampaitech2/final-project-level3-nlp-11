import os
import urllib.parse
import streamlit as st
import streamlit.components.v1 as components

from styles.main_list import main_list_style
from components.main_list_item import parser_data
from api.tour_api import Api
from api.model import ModelApi
from api.survey import PostSurvey
from dotenv import load_dotenv


env_path = os.path.expanduser("~/final-project-level3-nlp-11/code/.env")
load_dotenv(dotenv_path=env_path, verbose=True)


def info_page(query_params, cookie_manager):
    with st.spinner("찾는 중!..."):
        spot = query_params["target"][0]
        area = query_params["location"][0]
        user_query = query_params["query"][0]
        result_contents = ""

        result = Api().get_tour_topK([spot], area)
        result_contents = ModelApi().request_similar(spot, area)

        img_url = result[0]["img_url"]
        context = result[0]["context"]
        addr = result[0]["addr1"]
        tel = result[0]["tel"]
        context = context.replace("\n", "<br>")
        query = urllib.parse.quote_plus(addr + " " + spot)

        # TODO 번호 여러개인 경우 처리
        if tel == "전화 정보 없음":
            tmp = f"<b>Tel</b>: {tel}"
        else:
            tmp = f"""  <a href="tel:{tel}">
                            <b>Tel</b>: {tel}
                        </a>"""
        style_img = ""
        if img_url == "https://i.imgur.com/wbN96ze.png":
            style_img = "opacity:0.5;border:1px solid rgb(29,51,63,.4);padding-top:45px;padding-bottom:45px"
    html = f"""
            <div style="font-size:38px;font-weight:bold;clear:both;padding-top:50px;padding-left:20px"> {spot} </div>
            {main_list_style()}
            
            <div style="font-size:62.5%">
                <div style="height:100%;clear:both;padding:10px">
                    <div id=context_img>
                        <img src="{img_url}" width="100%" style="border-radius:10px;vertical-align:middle;{style_img}" >
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
            
            """

    html_list = f""" 
            <div style="font-size:38px;font-weight:bold;clear:both;padding-top:10px;padding-left:20px"> {area} 관련 추천 명소 </div>
            <div style="font-size:62.5%;">
                {parser_data(result_contents,area)}
            </div>
            """

    st.markdown(html, unsafe_allow_html=True)

    query_info = []
    if cookie_manager.get("query_info"):
        query_info = cookie_manager.get("query_info")
    else:
        info = []
        cookie_manager.set("query_info", info)

    def set_cookie(flag):
        query_info.append(user_query + spot)
        cookie_manager.delete("query_info")
        cookie_manager.set("query_info", query_info)
        PostSurvey().post_servey(user_query, area, spot, flag)

    # TODO  쿼리  존재 여부 확인 완료
    if not (user_query == "None" or user_query + spot in query_info):

        html_survey = f"""
            <div style="font-size:30px;margin:20px;text-align:center;height:100%">
                <b>"{user_query}"</b><br>에 대한 결과 <br><b>"{spot}"</b><br>은 도움이 되었나요?<br>
            </div>            
            """

        st.markdown(html_survey, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
        with col1:
            pass
        with col2:
            st.button("좋아요~!", on_click=set_cookie, kwargs=dict(flag=True))

        with col3:
            st.button("싫어요 ㅠ", on_click=set_cookie, kwargs=dict(flag=False))
        with col4:
            pass

    st.markdown(html_list, unsafe_allow_html=True)
