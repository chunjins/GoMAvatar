#!/bin/bash
conda activate GoMAvatar
cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code


id=$1

DATASET='mvhuman'
SUBJECTS=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')

SUBJECT=${SUBJECTS[id-1]}

#python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml
python eval.py --cfg exps/${DATASET}/${DATASET}_${SUBJECT}.yaml --type view