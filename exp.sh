#!/usr/bin/env bash
train=$1
exp_id=$2
exp_dir="/home/pirate03/Downloads/prediction_data/crop256/exp"
shopt -s nullglob
numdirs=($exp_dir/*/)
numdirs=${#numdirs[@]}
#numtrains=($exp_dir/train*)
#numtrains=${#numtrains[@]}
#numtests=($exp_dir/test*)
#numtests=${#numtests[@]}

case $train in
train)
    echo "train"
    is_train=True
    if [ -z $exp_id ]
    then
        exp_id=$numdirs
        mkdir ${exp_dir}/${exp_id}
    fi
;;
test)
    echo "test"
    is_train=False
    if [ -z $exp_id ]
    then
        exp_id=$(($numdirs-1))
    fi
;;
*)
    echo "wrong argument 1"
    exit
;;
esac

#if [ $train = "train" ]
#then
#    is_train=true
#    if [ -z $exp_id ]
#    then
#        exp_id=$(($numdirs+1))
#        mkdir ${exp_dir}/${exp_id}
#    fi
#else
#    is_train=false
#    if [ -z $exp_id ]
#    then
#        exp_id=$numdirs
#    fi
#fi

log_dir=${exp_dir}/${exp_id}
CUDA_VISIBLE_DEVICES=0 unbuffer python fcn_maxent_irl_car_sparse_feedback.py \
                                --log_dir $log_dir --$train > ${log_dir}/${train}.txt 2>&1 &
#echo "var is_train: $is_train"
#CUDA_VISIBLE_DEVICES=0 unbuffer python fcn_maxent_irl_car_sparse_feedback.py \
#                                --log_dir $log_dir --$train