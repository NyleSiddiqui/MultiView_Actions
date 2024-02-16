#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH -C gmem24
#SBATCH --job-name=ntu60cv
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
module list                            # Have Nvidia tell us the GPU/CPU mapping so we know
module load cuda/11.7
nvidia-smi topo -m

python3 main.py --train_classifier --gpu 0 --run_id ntu60cv --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

