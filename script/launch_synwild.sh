#!/bin/bash
conda activate GoMAvatar
cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code


id=$1

DATASET='synwild'
SUBJECTS=('00000_random' '00020_Dance' '00027_Phonecall' '00069_Dance' '00070_Dance')

SUBJECT=${SUBJECTS[id-1]}

python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml
#python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view