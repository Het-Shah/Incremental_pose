#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 153 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug_0.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 154 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug_1.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 155 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_0_5_aug.yaml > ../final_logs/rbf_dpp_0_5_aug_2.txt


CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 156 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug_0.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 157 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug_1.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 158 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_50_aug.yaml > ../final_logs/rbf_dpp_50_aug_2.txt


CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 159 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug_0.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 160 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug_1.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 161 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug_2.txt