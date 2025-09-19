#!/bin/bash
conda activate GoMAvatar
cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code


id=$1

DATASET='youtube'
SUBJECTS=('story' 'invisable' 'way2sexy')

SUBJECT=${SUBJECTS[id-1]}

python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml
python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose