#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 120 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 121 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 122 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug.txt

CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 129 --cfg /media2/het/Incremental_pose/animalpose/configs/icarl.yaml > ../final_logs/icarl.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 130 --cfg /media2/het/Incremental_pose/animalpose/configs/icarl.yaml > ../final_logs/icarl.txt
