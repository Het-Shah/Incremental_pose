#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python train_incremental_herding.py --exp-id 99 --cfg ./configs/finetuning.yaml > ./final_logs/finetuning_1.txt
CUDA_VISIBLE_DEVICES=1 nohup python train_incremental_herding.py --exp-id 100 --cfg ./configs/finetuning.yaml > ./final_logs/finetuning_2.txt
CUDA_VISIBLE_DEVICES=1 nohup python train_incremental_herding.py --exp-id 101 --cfg ./configs/finetuning.yaml > ./final_logs/finetuning_3.txt