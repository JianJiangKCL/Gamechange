#!/bin/bash

cd ..
python main.py -c configs/mnist_resnet9.yaml  --dataset_path /jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/mnist  --results_dir results/mnist -t lr=0.001 -t beta_f=100  -t out_planes=64 --grad_clip 0 -t epochs=200 --optimizer adam  #-t n_f_emb=64 -t n_dw_emb=64 -t n_pw_emb=64 -t epochs=40