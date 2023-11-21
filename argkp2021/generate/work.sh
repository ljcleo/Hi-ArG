#!/usr/bin/env bash

python 00_convert.py
python 10_parse.py
python 11_align.py
python 12_convert.py
python 20_merge.py
python 30_refine.py
python 40_combine.py
python 41_output.py
python 42_reindex.py
