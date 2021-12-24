#!/bin/bash

# data crawling
pip install python-dotenv==0.19.2
pip install selenium
pip install bs4
pip install tqdm
pip install git+https://github.com/haven-jeon/PyKoSpacing.git

# FE

pip install streamlit==1.1.0
pip install albumentations==1.1.0
pip install torch==1.10.0
pip install torchvision==0.11.1
pip install efficientnet-pytorch==0.7.1
pip install PyYAML==6.0
pip install streamlit-modal

# Model
pip install elasticsearch

## DAPT
pip install git+https://github.com/huggingface/transformers

## BE
pip install fastapi==0.70.0
pip install uvicorn==0.15.0
pip install google-cloud-storage==1.43.0
pip install google-api-python-client==2.33.0