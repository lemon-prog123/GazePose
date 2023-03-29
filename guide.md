# GazePose

## Training
For GazePose
```sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=9 #指定单卡训练

nohup python -u run.py #执行run.py 开始训练过程
--Gaussiandrop #Open Gaussian Temporal Dropout
--mesh         #Load Prior Data
--prior_head   #Use head pose prior
--prior_landmark #Use landmark prior
--backbone resnet18
--split 1 # The split for head pose
--dsplit 0.1 #The network split for Landmarks
--batch_size 64
--lr 1e-4
--checkpoint /data/shared/weixiyu/social-interactions/checkpoints/gaze360_model.pth #Load the pretrained gaze360 as pretrained model
--val_stride 13
--train_stride 3
--model GazePose
--num_workers 3
--train_path trainlog/output_gazeposeres18hubtrain #Create the trainlog path
--exp_path validationlog/output_gazeposeres18hub #Create the explog path
 >gazeposeres18hub.out &
```



For Baseline
```sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=9 #指定单卡训练

nohup python -u run.py #执行run.py 开始训练过程
--Gaussiandrop #执行Gaussiandrop
--backbone resnet18 
--batch_size 64
--lr 5e-5
--checkpoint /data/shared/weixiyu/social-interactions/checkpoints/gaze360_model.pth #Load pretrained gaze360
--val_stride 13
--train_stride 3
--model GazeLSTM
--num_workers 3
--train_path trainlog/output_gazeres18train
--exp_path validationlog/output_gazeres18
>gazeres18.out &
```

上述分别对应[posetrain.sh](posetrain.sh)和[basetrain.sh](basetrain.sh)，对应paper中`Table 1`关于Ego4D的实验

## Module Ablation
我们需要分别对两个模块进行消融，`Spatial Feature Refinement` 和 `Temporal Dynamic Refinement`，对应`Table 3`和`Table 4`.

### Spatial Feature Refinement
不使用这个模块，可以直接运行[basetrain.sh](basetrain.sh)去完成实验，Baseline中不带有此模块

### Temporal Dynamic Refinement
不使用这个模块，在`.sh`文件中取消`--Gaussiandrop`即可不使用该模块

### Prior Info
在`posetrain.sh`的基础上，我们可以对不同先验信息消融。`--prior_head`和`--prior_landmark`指定了两种先验信息的使用，不使用时可以取消参数选择。

`--split`可以指定头部欧拉角度的划分细粒度，默认为1代表一度一个query,`--dsplit`指定关于面部关键点的划分细粒度，默认边长0.1的正方形。


## Interference
以[poseval.sh](poseval.sh)为例

```sh
#!/bin/bash

export CUDA_VISIBLE_DEVICES=9

nohup python -u run.py
--Gaussiandrop
--mesh
--prior_head
--prior_landmark
--backbone resnet18
--split 1
--batch_size 64
--checkpoint /data/shared/weixiyu/social-interactions/validationlog/output_gazeposeres18coordmeshaugnet/checkpoint/best.pth
--val_stride 13
--train_stride 3
--model GazePose
--num_workers 3
--val #指定推理模式，可以选择--val或者--eval
--exp_path validationlog/output_gazeposeres18hubval
>gazeposeres18hubval.out &
```

基本参数与训练时保持一致，利用`--val`和`--eval`分别指定验证集和训练集推理。

其中`--eval`指定的测试集推理结果，需要去到`--exp_path`指定的`\pred`目录下，获取`pred.csv`，并利用根目录的[csv_to_json.py](csv_to_json.py)转换为json,按照提交提示提交至[evalAi](https://eval.ai/web/challenges/challenge-page/1624/my-submission) 平台评测。