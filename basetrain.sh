#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

nohup python -u run.py
--Gaussiandrop
--backbone resnet18
--split 1
--batch_size 64
--lr 5e-5
--checkpoint /data/shared/weixiyu/social-interactions/checkpoints/gaze360_model.pth
--val_stride 13
--train_stride 3
--model GazeLSTM
--num_workers 3
--train_path trainlog/output_gazeres18train
--exp_path validationlog/output_gazeres18
>gazeres18.out &