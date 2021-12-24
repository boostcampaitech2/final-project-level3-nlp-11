## Structure
├─FE
│  ├─src
│  │  │  app.py
│  │  │
│  │  ├─api
│  │  │      model.py
│  │  │      survey.py
│  │  │      tour_api.py
│  │  │
│  │  ├─components
│  │  │      header.py
│  │  │      main_list_item.py
│  │  │
│  │  ├─model
│  │  │      dense.py
│  │  │      elastic_search.py
│  │  │      model_index.py
│  │  │      retrieval_similar.py
│  │  │
│  │  ├─pages
│  │  │      info.py
│  │  │      main.py
│  │  │      similar.py
│  │  │
│  │  ├─styles
│  │  │      main_list.py
│  │  │
│  │  └─util
│  │          get_cookie.py
│  │          get_name_list.py
│  │          theme_name_tuple.py
│  │
│  └─static
│          logo.png


### How to use
streamlit config
```
#config.toml
[server]
port=6006
headless=true

[browser]
serverAddress = "http://seokam.asuscomm.com/"
serverPort = 8088
```


```
streamlit run ./code/FE/src/app.py 
```