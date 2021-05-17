#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 114 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50.yaml > ../final_logs/rbf_dpp_50.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 115 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50.yaml > ../final_logs/rbf_dpp_50.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 116 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50.yaml > ../final_logs/rbf_dpp_50.txt