#!/bin/bash

python run_mlm.py --model_name_or_path monologg/kobigbird-bert-base \
    --train_file ./DAPT_DOCS.txt \
    --max_seq_length 1024 \
    --output_dir ./output/batch_8 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --save_steps 7000 \
    --logging_steps 50\
    --do_train \
    --do_eval \
    --eval_step 7000\
    --evaluation_strategy steps
    --report_to wandb
    