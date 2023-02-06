#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH -C gmem24
#SBATCH --job-name=ntu60cv+v1
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
module list                            # Have Nvidia tell us the GPU/CPU mapping so we know
module load cuda/11.7
module load decord-cuda-11-7
nvidia-smi topo -m
#source activate /home/siddiqui/.conda/envs/test

python3 main.py --train_classifier --gpu 0 --run_id ntu60cv+v1 --run_description "train ViViT on PK" --dataset ntu_rgbd_60 --model_version 'v1' --input_type "rgb" --num_frames 16 --input_dim 224 --patch_size 16 --hidden_dim 256 --num_heads 8 --num_layers 2 --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM --f1_threshold 0.5 --skip 0 --steps 1 #--checkpoint /home/siddiqui/Action_Biometrics/results/saved_models/norm-mergedNTUPK+V3+R3DwActions_03-01-23_1336/model_5_70.4075.pth

#python3 dataloader.py
#python3 preprocessCharades.py
#python3 model.py
