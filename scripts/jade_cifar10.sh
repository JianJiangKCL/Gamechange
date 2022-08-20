#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=72:00:00

# set name of job
#SBATCH --job-name=gc_cifar10

# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jian.jiang@kcl.ac.uk

# Load the default environment
#module load nvidia/cuda-11.2
source ~/.bashrc
conda activate KM

cd ..
python main.py -c configs/cifar10_resnet9.yaml  --dataset_path /jmain02/home/J2AD019/exk04/jxj51-exk04/dataset/CIFAR10  --results_dir results/cifar10 -t lr=0.001 -t beta_f=100  -t out_planes=64 -t n_f_emb=512 -t n_dw_emb=512 -t n_pw_emb=512 -t epochs=100 -t batch_size=128 --scheduler warm_cosine  --distributed