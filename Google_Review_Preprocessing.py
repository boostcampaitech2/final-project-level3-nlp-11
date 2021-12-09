#!/usr/bin/env python
# coding: utf-8

# In[359]:


import pandas as pd
import numpy as np


# In[360]:


temp = pd.read_csv("./all_reviews.csv")


# In[361]:


temp


# In[362]:


temp.iloc[0]['review']


# In[363]:


preprocessed_review = temp['review']


# In[364]:


preprocessed_review


# ## HTML태그 제거

# In[365]:


import re
def remove_html(texts):
    """
    HTML 태그를 제거합니다.
    ``<p>안녕하세요 ㅎㅎ </p>`` -> ``안녕하세요 ㅎㅎ ``
    """
    preprcessed_text = []
    for text in texts:
        text = re.sub(r"<[^>]+>\s+(?=<)|<[^>]+>", "", text).strip()
        if text:
            preprcessed_text.append(text)
    return preprcessed_text


# In[366]:


preprocessed_review = remove_html(preprocessed_review)


# In[367]:


print(len(preprocessed_review))


# ## 이메일 제거

# In[368]:


def remove_email(texts):
    """
    이메일을 제거합니다.#웹에서 긁어 오면 이메일 같은 개인정보도 크롤링 하는 경우가 ㅠㅠ -> 반드시 제거해줘야
    ``홍길동 abc@gmail.com 연락주세요!`` -> ``홍길동  연락주세요!``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[369]:


preprocessed_review = remove_email(preprocessed_review)


# In[370]:


print(len(preprocessed_review))


# ## 해쉬태그 제거

# In[372]:


removed_index = []
def remove_hashtag(texts):
    """
    해쉬태그(#)를 제거합니다.
    ``대박! #맛집 #JMT`` -> ``대박!  ``
    """
    global removed_index 
    preprocessed_text = []
    for i, text in enumerate(texts):
        text = re.sub(r"#\S+", "", text).strip()
        if text:
            preprocessed_text.append(text)
        else:
            removed_index.append(i)
    return preprocessed_text


# In[373]:


preprocessed_review = remove_hashtag(preprocessed_review)


# In[374]:


print(len(preprocessed_review))


# In[375]:


print(removed_index)


# ## @태그 제거

# In[376]:


def remove_user_mention(texts):
    """
    유저에 대한 멘션(@) 태그를 제거합니다.
    ``@홍길동 감사합니다!`` -> `` 감사합니다!``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"@\w+", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[377]:


preprocessed_review = remove_user_mention(preprocessed_review)


# In[379]:


print(len(preprocessed_review))


# ## URL 태그 제거

# In[380]:


def remove_url(texts):
    """
    URL을 제거합니다.
    ``주소: www.naver.com`` -> ``주소: ``   #url도 개인정보, 중요한 주소를 포함할 가능성이 있다, 주소가 그리고 언어를 배우는데 도움이 되지는 않음
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
        text = re.sub(r"pic\.(\w+\.)+\S*", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[381]:


preprocessed_review = remove_url(preprocessed_review)


# In[382]:


print(len(preprocessed_review))


# ## 의미없는 문자들 제거

# In[383]:


def remove_bad_char(texts): #이건 한국어만의 문제인데, 가끔 한국어 크롤링을 하면 이와 같은 의미 없는 문자들이 크롤링 되는 경우가 ㅠㅠ
    """
    문제를 일으킬 수 있는 문자들을 제거합니다.
    """
    bad_chars = {"\u200b": "", "…": " ... ", "\ufeff": ""}
    preprcessed_text = []
    for text in texts:
        for bad_char in bad_chars:
            text = text.replace(bad_char, bad_chars[bad_char])
        text = re.sub(r"[\+á?\xc3\xa1]", "", text)
        if text:
            preprcessed_text.append(text)
    return preprcessed_text


# In[384]:


preprocessed_review = remove_bad_char(preprocessed_review)


# In[386]:


print(len(preprocessed_review))


# ## 저작권 관련 텍스트 제거

# In[387]:


def remove_copyright(texts):
    """
    리뷰 내 포함된 저작권 관련 텍스트를 제거합니다.
    ``(사진=저작권자(c) 연합뉴스, 무단 전재-재배포 금지)`` -> ``(사진= 연합뉴스, 무단 전재-재배포 금지)`` TODO 수정할 것
    """
    re_patterns = [
        r"\<저작권자(\(c\)|ⓒ|©|\(Copyright\)|(\(c\))|(\(C\))).+?\>",
        r"저작권자\(c\)|ⓒ|©|(Copyright)|(\(c\))|(\(C\))"
    ]
    preprocessed_text = []
    for text in texts:
        for re_pattern in re_patterns:
            text = re.sub(re_pattern, "", text).strip()
        if text:
            preprocessed_text.append(text)    
    return preprocessed_text


# In[388]:


preprocessed_review = remove_copyright(preprocessed_review)


# In[389]:


print(len(preprocessed_review))


# In[394]:


answer = pd.read_csv("./emergency.csv")


# In[395]:


answer


# ## 이미지에 대한 label 제거

# In[26]:


def remove_photo_info(texts):
    """
    리뷰 내 포함된 이미지에 대한 label을 제거합니다.
   
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"\(출처 ?= ?.+\) |\(사진 ?= ?.+\) |\(자료 ?= ?.+\)| \(자료사진\) |사진=.+기자 ", "", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[27]:


preprocessed_review = remove_photo_info(preprocessed_review)


# In[28]:


preprocessed_review


# ## 괄호 내부 전처리

# In[29]:


def remove_useless_breacket(texts):
    """
    괄호 내부에 의미가 없는 정보를 제거합니다.
    아무런 정보를 포함하고 있지 않다면, 괄호를 통채로 제거합니다.
    ``수학(,)`` -> ``수학``
    ``수학(數學,) -> ``수학(數學)``
    """
    bracket_pattern = re.compile(r"\((.*?)\)")
    preprocessed_text = []
    for text in texts:
        modi_text = ""
        text = text.replace("()", "")  # 수학() -> 수학
        brackets = bracket_pattern.search(text)
        if not brackets:
            if text:
                preprocessed_text.append(text)
                continue
        replace_brackets = {}
        # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
        # e.g. {'2,8': '(數學)','34,37': ''}
        while brackets:
            index_key = str(brackets.start()) + "," + str(brackets.end())
            bracket = text[brackets.start() + 1 : brackets.end() - 1]
            infos = bracket.split(",")
            modi_infos = []
            for info in infos:
                info = info.strip()
                if len(info) > 0:
                    modi_infos.append(info)
            if len(modi_infos) > 0:
                replace_brackets[index_key] = "(" + ", ".join(modi_infos) + ")"
            else:
                replace_brackets[index_key] = ""
            brackets = bracket_pattern.search(text, brackets.start() + 1)
        end_index = 0
        for index_key in replace_brackets.keys():
            start_index = int(index_key.split(",")[0])
            modi_text += text[end_index:start_index]
            modi_text += replace_brackets[index_key]
            end_index = int(index_key.split(",")[1])
        modi_text += text[end_index:]
        modi_text = modi_text.strip()
        if modi_text:
            preprocessed_text.append(modi_text)
    return preprocessed_text


# In[30]:


preprocessed_review = remove_useless_breacket(preprocessed_review)


# In[31]:


preprocessed_review


# ## 의미 없는 반복 줄이기

# In[32]:


get_ipython().system('pip install soynlp')


# In[33]:


from soynlp.normalizer import *


# In[34]:


def remove_repeat_char(texts):
    preprocessed_text = []
    for text in texts:
        text = repeat_normalize(text, num_repeats=2).strip() #ㅋㅋㅋㅋㅋㅋㅋㅋ 를 ㅋㅋ로
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[35]:


preprocessed_review = remove_repeat_char(preprocessed_review) 


# ## 스페이스 여러칸 한칸으로 줄이기

# In[36]:


def remove_repeated_spacing(texts): #전처리 하면서 여러개가 삭제되는데, 이때 space가 커질수가 있다. -> space 한칸으로 줄일 필요!!
    """
    두 개 이상의 연속된 공백을 하나로 치환합니다.
    ``오늘은    날씨가   좋다.`` -> ``오늘은 날씨가 좋다.``
    """
    preprocessed_text = []
    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[37]:


preprocessed_review = remove_repeated_spacing(preprocessed_review)


# In[38]:


preprocessed_review


# ## 띄어쓰기 안되어 있는 문장 띄어쓰기 처리

# In[39]:


get_ipython().system('pip install git+https://github.com/haven-jeon/PyKoSpacing.git ')


# In[40]:


from tqdm import tqdm


# In[41]:


from pykospacing import Spacing


# In[42]:


spacing = Spacing()


# In[59]:


def spacing_sent(texts):
    """
    띄어쓰기를 보정합니다.
    """
    preprocessed_text = []
    for text in tqdm(texts, mininterval=100):
        text = spacing(text)
        if text:
            preprocessed_text.append(text)
    return preprocessed_text


# In[57]:


print(spacing_sent(["안녕하세요저는로봇입니다", "너는뭐하는인간이냐"]))


# In[60]:


preprocessed_review = spacing_sent(preprocessed_review)


# ## 전처리를 진행하다, 중간에 없어진 칸들이 있는데, 이에 대한 처리를 진행합니다

# In[576]:


final_prev = preprocessed_review
final_prev = pd.DataFrame(final_prev)
final_prev.to_csv('./final_prev.csv')


# In[ ]:


print(removed_index)


# In[ ]:


for r_i in removed_index:
    final_prev.insert(r_i, "")


# In[ ]:


final_review = pd.DataFrame(final_prev)


# In[ ]:


final_answer = pd.read_csv("./all_reviews.csv")


# In[ ]:


import pandas as pd
final_answer = pd.concat([final_answer, final_review], axis=1) 


# In[ ]:


final_answer = final_answer.rename(columns={0: 'new_review'})


# In[ ]:


final_answer.to_csv("final_review.csv")


# In[61]:


'''
emergency = preprocessed_review
for_emergency = pd.DataFrame(emergency)
for_emergency.to_csv('./emergency.csv')
'''


# In[62]:


#emergency


# 653392

# In[76]:


'''
compare = pd.read_csv("./all_reviews.csv")
compare = compare['review']
print(len(compare))
print(len(emergency))
'''


# In[81]:


#compare = list(compare)    


# 3405

# In[402]:


#answer = answer['0']


# In[404]:


#answer_change = list(answer)


# In[405]:


#answer_change


# In[415]:


#print(removed_index)


# In[556]:


#i = 645329
#print(compare[i])
#print(answer_change[i])


# In[557]:


#print(len(answer_change))


# In[548]:


#answer_change.insert(i, "")


# In[547]:


#for j in range(i, len(compare)):
#    if abs(len(compare[j]) - len(answer_change[j])) >= 15:
#        print(j)


# In[495]:


#removed_index = removed_index[21:]


# In[501]:


#for r_i in removed_index:
#    answer_change.insert(r_i, "")


# In[497]:


#removed_index


# In[488]:


#answer_change.insert(i, "")


# In[558]:


#changed_review = answer_change


# In[559]:


#print(len(changed_review))


# In[562]:


#changed_review = pd.DataFrame(changed_review)


# In[566]:


#changed_review


# In[560]:


#final_answer = pd.read_csv("./all_reviews.csv")


# In[569]:



#import pandas as pd
#final_answer = pd.concat([final_answer, changed_review], axis=1) 


# In[570]:


#final_answer


# In[573]:


#final_answer = final_answer.rename(columns={0: 'new_review'})


# In[574]:


#final_answer


# In[575]:


#final_answer.to_csv("final_review.csv")

