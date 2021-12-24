## 여행지 추천 웹 서비스
- 입력 문장에 대한 관광지 추천
- 특정 관광지와 유사한 다른 장소 추천
- 지역 기반 추천

![](https://i.imgur.com/10i8erb.png)

### 개발 문서
[개발 & 실험 & 회의록 링크](https://github.com/boostcampaitech2/final-project-level3-nlp-11/wiki)

### Environments
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

### Dependencies
- python-dotenv==0.19.2
- selenium
- bs4
- tqdm==4.62.3
- PyKoSpacing
- streamlit==1.1.0
- albumentations==1.1.0
- torch==1.10.0
- torchvision==0.11.1
- efficientnet-pytorch==0.7.1
- PyYAML==6.0
- streamlit-modal
- elasticsearch
- transformers==4.14.1
- fastapi==0.70.0
- uvicorn==0.15.0
- google-cloud-storage==1.43.0
- google-api-python-client==2.33.0

### Structure
   
<details>
<summary> See Structure </summary>
    
    
```
root
│  .gitignore
│  README.md
│
├─.github
│  │  PULL_REQUEST_TEMPLATE.md
│  │
│  └─ISSUE_TEMPLATE
│          issue.md
│
├─code
│  │  .env.template
│  │
│  ├─BE
│  │      dense.py
│  │      logger.py
│  │      main.py
│  │      pyproject.toml
│  │      __init__.py
│  │
│  ├─DATA_GEN
│  │  │  main.py
│  │  │  split_chunks.py
│  │  │  val_query.py
│  │  │  __init__.py
│  │  │
│  │  ├─config
│  │  │      config.conf
│  │  │
│  │  └─src
│  │      │  Cafe.ipynb
│  │      │
│  │      ├─crawling
│  │      │  │  blog_crawling.py
│  │      │  │  cafe_crawling.py
│  │      │  │  scrapper.py
│  │      │  │  __init__.py
│  │      │  │
│  │      │  └─utils
│  │      │          blog_parser.py
│  │      │          csv_to_json.py
│  │      │          replace_text.py
│  │      │          request_blog.py
│  │      │          spacing.py
│  │      │          __init__.py
│  │      │
│  │      ├─mk_pair
│  │      │      mk_pair.py
│  │      │      __init__.py
│  │      │
│  │      └─preprocess
│  │              preprocess_blog.py
│  │              preprocess_google.py
│  │              __init__.py
│  │
│  ├─FE
│  │  ├─src
│  │  │  │  app.py
│  │  │  │
│  │  │  ├─api
│  │  │  │      model.py
│  │  │  │      survey.py
│  │  │  │      tour_api.py
│  │  │  │
│  │  │  ├─components
│  │  │  │      header.py
│  │  │  │      main_list_item.py
│  │  │  │
│  │  │  ├─model
│  │  │  │      dense.py
│  │  │  │      elastic_search.py
│  │  │  │      model_index.py
│  │  │  │      retrieval_similar.py
│  │  │  │
│  │  │  ├─pages
│  │  │  │      info.py
│  │  │  │      main.py
│  │  │  │      similar.py
│  │  │  │
│  │  │  ├─styles
│  │  │  │      main_list.py
│  │  │  │
│  │  │  └─util
│  │  │          get_cookie.py
│  │  │          get_name_list.py
│  │  │          theme_name_tuple.py
│  │  │
│  │  └─static
│  │          logo.png
│  │
│  └─MODEL
│      ├─DAPT
│      │      DAPT.sh
│      │      json_data_to_text.py
│      │      requirements.txt
│      │      run_mlm.py
│      │
│      ├─dense
│      │      dataset.py
│      │      retrieval.py
│      │      utils.py
│      │
│      └─sparse
│              elastic_search.py
│              elastic_search.sh
│
├─etc
│      my_stop_dic.txt
│
├─install
│      install_requirements.sh
│
└─sparse
        retrieval.py
        retrieval_similar.py
```

</details>


### How to use
#### Install requirements
```
bash ./install/install_requirements.sh
```

#### BE (FastAPI)
[Detail](https://github.com/boostcampaitech2/final-project-level3-nlp-11/tree/master/code/BE)

#### FE (streamlit)
[Detail](https://github.com/boostcampaitech2/final-project-level3-nlp-11/tree/master/code/FE)

#### DATA_gen
[Detail](https://github.com/boostcampaitech2/final-project-level3-nlp-11/tree/master/code/DATA_GEN)

#### MODEL
[Detail](https://github.com/boostcampaitech2/final-project-level3-nlp-11/tree/master/code/MODEL)

### Members of Team AI-ESG
| Name | github | contact |
| -------- | -------- | -------- |
| 문석암     | [Link](https://github.com/mon823) | mon823@naver.com |
| 박마루찬 | [Link](https://github.com/MaruchanPark) | shaild098@naver.com |
| 박아멘 | [Link](https://github.com/AmenPark) | puzzlistpam@gmail.com |
| 우원진 | [Link](https://github.com/woowonjin) | dndnjswls613@naver.com |
| 윤영훈 | [Link](https://github.com/wodlxosxos) | wodlxosxos73@gmail.com |
| 장동건 | [Link](https://github.com/mycogno) | jdg4661@gmail.com |
| 홍현승 | [Link](https://github.com/Hong-Hyun-Seung) | honghyunseung100@gmail.com |