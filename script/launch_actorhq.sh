#!/bin/bash
conda activate GoMAvatar

cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code

id=4

DATASET='actorhq'
SUBJECTS=('actor0101' 'actor0301' 'actor0601' 'actor0701' 'actor0801')

SUBJECT=${SUBJECTS[id-1]}

python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml
python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view
python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose
