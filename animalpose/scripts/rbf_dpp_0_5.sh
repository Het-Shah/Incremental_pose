#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 111 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5.yaml > ../final_logs/rbf_dpp_0_5.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 112 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5.yaml > ../final_logs/rbf_dpp_0_5.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 113 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5.yaml > ../final_logs/rbf_dpp_0_5.txt