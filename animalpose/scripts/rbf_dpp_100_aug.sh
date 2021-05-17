#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 126 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 127 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 128 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100_aug.yaml > ../final_logs/rbf_dpp_100_aug.txt

CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 131 --cfg /media2/het/Incremental_pose/animalpose/configs/icarl.yaml > ../final_logs/icarl.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 131 --cfg /media2/het/Incremental_pose/animalpose/configs/eeil.yaml > ../final_logs/eeil.txt