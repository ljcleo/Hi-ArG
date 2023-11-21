#!/usr/bin/env bash

pae() {
    echo ${@}
    ${@}
}

seed=${1}
mode=${2}
comment=${3}
cpStart=${4}
cpEnd=${5}
format=${6}
cuda=${7}

if [[ "${format}" == "" ]]; then
    format='sc pl cl'
fi

if [[ "${cuda}" != "" ]]; then
    export CUDA_VISIBLE_DEVICES=${cuda}
fi

for fm in ${format}; do
    if [[ "${fm}" == cl ]]; then
        bs=8
    else
        bs=32
    fi

    args="-s ${seed} -t ${bs} -d ${bs} -g 1 ${fm} ${mode}"

    if [[ "${cpStart}" == 0 ]]; then
        if [[ "${comment}" == 0 ]]; then
            pae python train.py ${args}
        fi

        cpStart=1
    fi

    for cp in $(seq ${cpStart} ${cpEnd}); do
        pae python train.py -p argsme -m ${comment} -c ${cp}0000 ${args}
    done
done
