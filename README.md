# GazePose

## Requiments
- pytorch
- opencv
- mediapipe
- numpy



## Quick Start

### Data preparation
You should prepare your data just as [*Ego4D*](https://github.com/EGO4D/social-interactions/tree/lam).
Specially, your structure should be like this:

```
data/
* csv/
  
* json/
  
* split/
  
* result_LAM/
  
* json_original/

* video_imgs/
  
* face_imgs/
  * uid
    * trackid
      * face_xxxxx.jpg
```

It should be noticed that pictures in face_imgs are preprocessed from video_imgs.
We simply crop the face bbox and resize every face as (224,224), re-organizing them as uid/trackid/face_xxxxx.jpg, where xxxxx means its frameid.
You can see the detail in [dataset/data_loader.py](./dataset/data_loader.py).

And for the annations of headpose, you can directly use model in [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) and run it on face crops.
We use 'uid-trackid-frameid' as keys in our headpose annations. Please refer to [dataset/data_loader.py](./dataset/data_loader.py). And we provide the code to predict head pose for Ego4D in [headpose.py](headpose.py).

For the annations of facial landmarks, we use model in [mediapipe](https://github.com/google/mediapipe) and run it on face crops. And we provide the code to predict landmarks for Ego4D in [meshp.py](meshp.py).

For the Gaussian Temporal Dropout, we can obtain images' quality through [vartool.py](vartool.py) and also can obtain the distribution for each split. 
We release our filter JSON FILE for train set. [train_filter](https://www.aliyundrive.com/s/cfQFkMvkzUX) Please download it and change the value of "filter_file" in [config.py](common/config.py).

Please use above code and files to obtain the prior json files, put them in the same directory and change the value "prior_path" in [config.py](common/config.py).
### Train
Specify the arguments listed in [common/config.py](./common/config.py) if you want to customize the training.

You can use:
```
bash posetrain.sh
```
To simply start a training process, it will run GazePose training process on your first GPU.

For the baseline, you can use:
```
bash basetrain.sh
```

### 3. Inference
You can use:
```
bash poseval.sh
bash baseval.sh
```
To simply start a inference process, specific ```--checkpoint [your checkpoint]``` to choose your checkpoint for evaluate on GazePose or Baseline. We provide a pretrained checkpoint for GazePose. [model](https://disk.pku.edu.cn:443/link/42B9E356F3D182E30CC96342A2DBB98E)
The argument ```--val``` will start a evaluate on validation dataset and the argument ```--eval``` will start a evaluate on test dataset.

Specially, the test set of [*Ego4D*](https://github.com/EGO4D/social-interactions/tree/lam) should be placed as follows:
```
videos_challenge/
* uid
...

GazePose/
* data
...
```

Or you can also specified the "test_path" in [config.py](common/config.py).