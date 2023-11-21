#!/usr/bin/env bash

ln -sf ../prepare/data/prepared ./data
ln -sf ../prepare/tokenizer ./tokenizer

python cache.py
python get_cache_info.py
