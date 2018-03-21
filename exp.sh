#!/usr/bin/env bash
exp_dir="/home/pirate03/Downloads/prediction_data/crop256/exp"
shopt -s nullglob
numdirs=($exp_dir/*/)
numdirs=${#numdirs[@]}
exp_id=$(($numdirs+1))
log_dir=${exp_dir}/${exp_id}
mkdir $log_dir
CUDA_VISIBLE_DEVICES=0 unbuffer python fcn_maxent_irl_car_sparse_feedback.py \
                                --log_dir $log_dir > ${log_dir}/test.txt 2>&1 &