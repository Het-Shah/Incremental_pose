#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 nohup python train_incremental_herding.py --exp-id 96 --cfg ./configs/oracle.yaml > ./final_logs/oracle_1.txt
CUDA_VISIBLE_DEVICES=0 nohup python train_incremental_herding.py --exp-id 97 --cfg ./configs/oracle.yaml > ./final_logs/oracle_2.txt
CUDA_VISIBLE_DEVICES=0 nohup python train_incremental_herding.py --exp-id 98 --cfg ./configs/oracle.yaml > ./final_logs/oracle_3.txt