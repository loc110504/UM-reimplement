#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# Sửa các thông số theo nhu cầu của bạn
dataset='pascal'
method='unimatch'
exp='r101'
split='1464'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python $method.py \
    --config=$config \
    --labeled-id-path $labeled_id_path \
    --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path 2>&1 | tee $save_path/$now.log
