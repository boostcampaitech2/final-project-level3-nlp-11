import sys
import os
import pickle
import argparse
import json

# import torch

from transformers import (
    AutoTokenizer,
)

from dense_model import BertEncoder

test_encoder = BertEncoder.from_pretrained('klue/bert-base').cuda()
test_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

test_input = '이것은 버트의 출력이 무엇인지 알아보기 위해 입력하는 문장'
tokenized = test_tokenizer(test_input)

test_encoder(tokenized)
# print(test_tokenizer.decode(tokenized['input_ids']))