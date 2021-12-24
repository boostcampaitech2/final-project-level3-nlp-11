## Structure
```
├─DATA_GEN
│  │  main.py
│  │  split_chunks.py
│  │  val_query.py
│  │  __init__.py
│  │
│  ├─config
│  │      config.conf
│  │
│  └─src
│      │  Cafe.ipynb
│      │
│      ├─crawling
│      │  │  blog_crawling.py
│      │  │  cafe_crawling.py
│      │  │  scrapper.py
│      │  │  __init__.py
│      │  │
│      │  └─utils
│      │          blog_parser.py
│      │          csv_to_json.py
│      │          replace_text.py
│      │          request_blog.py
│      │          spacing.py
│      │          __init__.py
│      │
│      ├─mk_pair
│      │      mk_pair.py
│      │      __init__.py
│      │
│      └─preprocess
│              preprocess_blog.py
│              preprocess_google.py
│              __init__.py
```
### How to use

fix the config
```
config path = ./code/DATA_GEN/config/config.conf

config default values

[crawling]

--start_state=서울
--start_type=관광지
--start_location=간데메공원
--use_start_point= True
--driver_path=src/crawling/utils/chromedriver 
--num=7 

[file_name]

--name_path=tour_spot_name.json
--info_path=info.json
--result_path=result.json
--csv_file_name=all_reviews_not_preprocessed.csv
--output_json_file_name=review.json
--output_result=result_prepro.json
--output_info=info_prepro.json
--output_google=google_prepro.json
--output_pair=pair.json
--output_pair_info=pair_info.json

[preprocess]

--minimum_blog_reviews=100
--minimun_blog_length=100
--minimun_google_rating=3
--useing_ko_spacing=T

[mkpair]
--minimun_pair_num=10
--clean_tokenizer=T

[progress]; 작성시 True
--run_crawling= 
--run_preprocess=
--run_mk_pair=T
```
```
python ./code/DATA_GEN/main.py
```
make validation set
```
python ./code/DATA_GEN/val_query.py
```