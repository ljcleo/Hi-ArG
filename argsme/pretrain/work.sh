#!/usr/bin/env bash

pae() {
    echo ${@}
    ${@}
}

seed=${1}
mode=${2}
comment=${3}
continue=${4}
ablation=${5}

if [[ "${6}" != "" ]]; then
    export CUDA_VISIBLE_DEVICES=${5}
fi

if [[ "${mode}" == 1 ]]; then
    dbs=16
else
    dbs=8
fi

if [[ "${continue}" == 1 ]]; then
    norest="-c"
else
    norest=""
fi

pae python train.py \
    ${ablation} \
    -t 32 \
    -d ${dbs} \
    -g 1 \
    -s ${seed} \
    -n ${norest} \
    argsme ${mode} ${comment}
