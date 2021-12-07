#/bin/bash

# python3 dense_inference.py --load_path_q encoder/q_encoder_4 --pickle_path ../data/dense/dense_embedding.bin_epoch_4.bin \
python3 dense_inference.py --load_path_q encoder/q_encoder_8 --pickle_path ../data/dense/dense_embedding.bin_epoch_8.bin > epoch_8.txt 

python3 dense_inference.py --load_path_q encoder/q_encoder_12 --pickle_path ../data/dense/dense_embedding.bin_epoch_12.bin > epoch_12.txt 

python3 dense_inference.py --load_path_q encoder/q_encoder_16 --pickle_path ../data/dense/dense_embedding.bin_epoch_16.bin > epoch_16.txt 