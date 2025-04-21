#!/bin/bash
conda activate GoMAvatar

cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code


id=$1

DATASET='mpi'
#SUBJECTS=('0056' 'FranziRed' 'Antonia' 'Magdalena')
SUBJECTS=('0056' 'FranziRed')


SUBJECT=${SUBJECTS[id-1]}

#python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml --resume
#python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view
python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose
