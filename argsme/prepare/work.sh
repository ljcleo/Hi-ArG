#!/usr/bin/env bash

mkdir -p data && cd data
ln -sf ../../../public/vocab ./vocab

mkdir -p dataset && cd dataset
ln -sf ../../../generate/data/final ./argsme

cd ../..

python 00_pair.py
python 10_vocab.py
python 11_tokenizer.py
python 20_node_edge_tokens.py
python 21_top_tokens.py
python 22_aligns.py
python 30_split.py
python 31_node_text_align.py
python 40_simplify.py
python 50_prepare.py
python 51_merge.py
python 52_mix.py
