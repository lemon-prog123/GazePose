#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

nohup python -u run.py   --Gaussiandrop  --backbone resnet18 --split 1 --batch_size 64 --num_encoder_layers 6 --num_decoder_layers 6 --dim_feedforward 2048 --val_stride 13  --train_stride 3  --model GazeLSTM --num_workers 3  --val --train_path trainlog/output_gazeres18trainval  --exp_path validationlog/output_gazeres18val >gazeres18val.out &