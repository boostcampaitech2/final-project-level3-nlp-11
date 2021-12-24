## Structure

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


### How to use

#### Dense

```
python ./code/MODEL/dense/retrieval.py --batch_size 4 \
    --run_name DAPT_val \
    --dataset_path /opt/ml/final-project-level3-nlp-11/data/DAPT_Val \
    --save_path_q ./encoder/q_encoder \
    --save_path_p ./encoder/p_encoder \
    --tokenizer_name monologg/kobigbird-bert-base \
    --token_length 1024 \
    --num_train_epochs 20 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --random_seed 211 \
    --p_enc_name_or_path monologg/kobigbird-bert-base \
    --q_enc_name_or_path monologg/kobigbird-bert-base \
    --save_pickle_path /opt/ml/final-project-level3-nlp-11/data/MODEL/dense_embedding.bin \
    --log_step 50 \
    --val_start_epoch 1 \
    --adam_epsilon 1e-8 \
    --gradient_accumulation_steps 1 \
    --warmup_steps 0
```

#### DAPT

make train data 
```
python ./code/MODEL/DAPT/json_data_to_text.py
```

run DAPT
```
# DAPT PATH = ./code/MODEL/DAPT
python ./code/MODEL/DAPT/run_mlm.py --model_name_or_path monologg/kobigbird-bert-base \
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
```