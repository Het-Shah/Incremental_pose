#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 105 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_1.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 106 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_2.txt
CUDA_VISIBLE_DEVICES=1 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 107 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_3.txt