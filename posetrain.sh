#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

nohup python -u run.py
--Gaussiandrop
--prior_head
--prior_landmark
--mesh
--backbone resnet18
--split 1
--batch_size 64
--lr 1e-4
--checkpoint /data/shared/weixiyu/social-interactions/checkpoints/gaze360_model.pth
--val_stride 13
--train_stride 3
--model GazePose
--num_workers 3
--train_path trainlog/output_gazeposeres18coordmeshaugnethubtrain
--exp_path validationlog/output_gazeposeres18coordmeshaugnethub
 >gazeposeres18coordmeshaugnethub.out &