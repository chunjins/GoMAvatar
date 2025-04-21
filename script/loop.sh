#!/bin/bash
conda activate GoMAvatar

cd /ubc/cs/home/c/chunjins/chunjin_shield/project/gomavatar/code

DATASET='actorhq'
subjects=('actor0101' 'actor0301' 'actor0601' 'actor0701' 'actor0801')

for SUBJECT in "${subjects[@]}"; do
  #python train.py --cfg exps/${DATASET}/${SUBJECT}.yaml --resume
  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view
  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose
  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_view
  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_pose
done

#DATASET='mvhuman'
#subjects=('100846' '100990' '102107' '102145' '103708' '200173' '204112' '204129')
#for SUBJECT in "${subjects[@]}"; do
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_view
#done
#
#DATASET='mpi'
#subjects=('Antonia' 'Magdalena')
#for SUBJECT in "${subjects[@]}"; do
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_pose
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_training
#done
#
#DATASET='mpi'
#subjects=('0056' 'FranziRed')
#for SUBJECT in "${subjects[@]}"; do
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type view
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type pose
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_view
#  python eval.py --cfg exps/${DATASET}/${SUBJECT}.yaml --type mesh_novel_pose
#done

