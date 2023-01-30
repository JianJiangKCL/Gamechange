#!/bin/bash

cd ..
CUDA_VISIBLE_DEVICES=1 python main.py -c configs/mnist_resnet9.yaml   --dataset_path /vol/jj/dataset/MNIST  --results_dir results/mnist