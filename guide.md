# GazePose


## Data Pre-processing
首先参照[Ego4D官方教程](https://github.com/EGO4D/social-interactions/blob/lam/README.md) ,对原始文件进行如下组织

data/
* csv/
  * manifest.csv
* json/
  * av_train.json
  * av_val.json
* split/
  * train.list
  * val.list
* videos/
  * 00407bd8-37b4-421b-9c41-58bb8f141716.mp4
  * 007beb60-cbab-4c9e-ace4-f5f1ba73fccf.mp4
  * ...

接下来执行官方提供的.sh文件进行数据预处理
```
bash scripts/extract_frame.sh
python scripts/preprocessing.py
```
你将会得到video_imgs文件夹，其中放有数据集中每一帧图片，但是还未处理成人脸截图。为了使用方便，你可以选择将每张人脸视为可被如下三元组唯一标记:
```
[uid-trackid-frameid]
```
其中uid代表视频id，trackid是在经过人脸追踪后分段的人脸Track ID，frameid即是对应原始视频中的帧号，我们按照这种三元组重新组织整个图片目录:

```
*data/
  * face_imgs/
    * uid
      * trackid
        * face_xxxxx.jpg
```
其中每一张图片都是根据标注的[bbox](dataset/data_loader.py)截取的224*224的人脸截图。

其次，需要修改 [config.py](common/config.py)，默认目录都为./data/


对于先验数据，我们有三个部分的预处理先验数据，分别是头部姿势，面部关键点，时间轴上的质量选择先验

我们利用了[deep-head-pose](https://github.com/natanielruiz/deep-head-pose)作为头部姿势检测模型，对于每一张人脸图片，其给出了偏转角，俯仰角，翻滚角的预测，具体操作流程可以参考 [headpose.py](headpose.py).

我们利用了[mediapipe](https://github.com/google/mediapipe)作为面部关键点检测模型，对于每一张人脸图片，其给出了左右眼，鼻子，左右嘴唇的预测，具体操作流程可以参考 [mesh.py](mesh.py).

对于人脸质量评估，我们对数据集的三个划分都进行了质量评估，并且拟合了高斯分布，具体参考 [vartool.py](vartool.py).


对于训练集，验证集和测试集，我们都需要给出其先验文件，我们在[阿里云盘](https://www.aliyundrive.com/s/15uuWKE6gWc) 给出了这些先验文件，可以直接使用，当获取所有先验文件以后，将其放入./data/json_prior目录。

data/
* csv/
  * manifest.csv
* json/
  * av_train.json
  * av_val.json
* split/
  * train.list
  * val.list
* videos/
  * 00407bd8-37b4-421b-9c41-58bb8f141716.mp4
  * 007beb60-cbab-4c9e-ace4-f5f1ba73fccf.mp4
  * ...
* json_prior/
  *train_mesh.json
  *val_mesh.json
  *test_mesh.json
  *train_headpose.json
  *...
* video_imgs/
  * face_imgs/
    * uid
      * trackid
        * face_xxxxx.jpg
  

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