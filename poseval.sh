#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

nohup python -u run.py
--Gaussiandrop
--mesh
--backbone resnet18
--split 1
--batch_size 64
--checkpoint /data/shared/weixiyu/social-interactions/validationlog/output_gazeposeres18coordmeshaugnet/checkpoint/best.pth
--val_stride 13
--train_stride 3
--model GazePose
--num_workers 3
--eval
--exp_path validationlog/output_gazeposeres18coordmeshaugnethubval
>gazeposeres18coordmeshaugnethubval.out &