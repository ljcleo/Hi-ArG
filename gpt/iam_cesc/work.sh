#!/usr/bin/env bash

apiKey=${1}

ln -sf ../../iam_cesc/generate/data/raw ./raw
python convert.py

for prompt in direct explain; do
    python predict.py ${prompt} ${apiKey}
done

python evaluate.py
