#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 117 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100.yaml > ../final_logs/rbf_dpp_100.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 118 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100.yaml > ../final_logs/rbf_dpp_100.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 119 --cfg /media2/het/Incremental_pose/animalpose/configs/rbf_dpp_100.yaml > ../final_logs/rbf_dpp_100.txt