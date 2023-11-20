#!/usr/bin/env bash

pae() {
    echo ${@}
    ${@}
}

. '/usr/local/miniconda3/etc/profile.d/conda.sh'
conda activate neural

seed=${1}
mode=${2}
comment=${3}
cpStart=${4}
cpEnd=${5}
cuda=${6}

if [[ "${cuda}" != "" ]]; then
    export CUDA_VISIBLE_DEVICES=${cuda}
fi

args="-s ${seed} -t 128 -d 64 -g 1 ${mode}"

if [[ "${cpStart}" == 0 ]]; then
    if [[ "${comment}" == 0 ]]; then
        pae python train.py ${args}
    fi

    cpStart=1
fi

for cp in $(seq ${cpStart} ${cpEnd}); do
    pae python train.py -p argsme -m ${comment} -c ${cp}0000 ${args}
done
