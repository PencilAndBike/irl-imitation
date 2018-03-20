#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 unbuffer python fcn_maxent_irl_car_sparse_feedback.py >test4.txt 2>&1 &