#!/bin/bash
#python train_PQ.py --lr 0.01 --batch_size 32 --n_dw_emb 32 --n_pw_emb 32 --n_f_emb 16 --dataset mnist  --dataset_path /jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/mnist --diff-cof 0.01 --result_dir results/test

python train_PQ.py --lr 0.01 --batch_size 32 --n_dw_emb 32 --n_pw_emb 32 --n_f_emb 32 --out_planes 3 --dataset mnist  --dataset_path /jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/mnist --diff-cof 0.01 --result_dir results/test