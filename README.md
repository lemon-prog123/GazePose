

### Data preparation
Switch to [*Ego4D*](https://github.com/EGO4D/social-interactions/tree/lam).

You should prepare your data just as Ego4D.

Specially, your structure should be like this:

data/
* csv/
  
* json/
  
* split/
  
* result_LAM/
  
* json_original/
  
* face_imgs/
  * uid
    * trackid
      * face_xxxxx.jpg


It should be noticed that pictures in face_imgs are preprocessed from video_imgs.
We simply crop the face bbox and resize every face as (224,224), re-organizing them as uid/trackid/face_xxxxx.jpg, where xxxxx means its frameid.
You can see the detail in [dataset/data_loader.py](./dataset/data_loader.py).

And for the annations of headpose, you can directly use model in [deep-head-pose](https://github.com/natanielruiz/deep-head-pose) and run it on your face crops.
We use 'uid-trackid-frameid' as keys in our headpose annations. Please refer to [dataset/data_loader.py](./dataset/data_loader.py).

And we release our filter JSON FILE for train set later in [train_filter](https://www.aliyundrive.com/s/cfQFkMvkzUX)

### Train

Specify the arguments listed in [common/config.py](./common/config.py) if you want to customize the training.

You can use:

```
bash posetrain.sh
```
To simply start a training process, it will run on your first GPU.


### 3. Inference
You can use:

```
bash poseval.sh
```
To simply start a inference process, specific ```--checkpoint [your checkpoint]``` to choose your checkpoint for evaluate.


