#!/usr/bin/env bash

ln -sf ../../argsme/prepare/tokenizer ./tokenizer

mkdir -p data && cd data
ln -sf ../../../public/vocab ./vocab

mkdir -p dataset/argkp && cd dataset/argkp
ln -sf ../../../../generate/data/task/*.jsonl ./
ln -sf ../../../../generate/data/final/*.bin ./

cd ../../..

python 0_node_edge_tokens.py
python 1_top_tokens_aligns.py
python 2_tasks.py
python 3_simplify.py
python 4_generate.py
