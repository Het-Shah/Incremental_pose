#!/bin/bash

CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 108 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 109 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp.txt
CUDA_VISIBLE_DEVICES=2 nohup python /media2/het/Incremental_pose/animalpose/train_incremental_herding.py --exp-id 110 --cfg /media2/het/Incremental_pose/animalpose/configs/dpp.yaml > ../final_logs/dpp.txt