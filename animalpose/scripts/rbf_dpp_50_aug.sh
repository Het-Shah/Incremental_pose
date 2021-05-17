#!/bin/bash

# CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 123 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 124 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 125 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug.txt

CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 129 --cfg /media2/het/Incremental_pose/animalpose/configs/eeil.yaml > ../final_logs/eeil.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 130 --cfg /media2/het/Incremental_pose/animalpose/configs/eeil.yaml > ../final_logs/eeil.txt
