#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 168 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_1.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 169 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_2.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 170 --cfg /media2/het/Incremental_pose/animalpose/configs/random.yaml > ../final_logs/random_3.txt


CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 171 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_1.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 172 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_2.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 173 --cfg /media2/het/Incremental_pose/animalpose/configs/herding.yaml > ../final_logs/herding_3.txt


CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 174 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp_1.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 175 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp_2.txt
CUDA_VISIBLE_DEVICES=0 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 176 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp_3.txt