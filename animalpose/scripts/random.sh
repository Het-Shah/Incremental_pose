#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 102 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_1.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 103 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_2.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 104 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_3.txt