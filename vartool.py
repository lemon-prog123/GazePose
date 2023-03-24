
import cv2
import os
from dataset.data_loader import ImagerLoader, TestImagerLoader
from common.utils import PostProcessor, get_transform, save_checkpoint, TestPostProcessor
from common.config import argparser
import json
import numpy as np
from scipy.stats import norm

def process(imgs,kframes,face_path):
    var_dict={}
    for index in range(len(kframes)):
        if index%10000==0:
            print(index,len(kframes))
        uid, trackid, frameid, _, label = imgs[kframes[index]]
        #print(len(imgs))
        uid_path=os.path.join(face_path,uid)
        track_path=os.path.join(uid_path,trackid)
        face_img=f'{track_path}/face_{frameid:05d}.jpg'
        if not os.path.exists(face_img):
            continue
        
        img = cv2.imread(face_img)
        imageVar=cv2.Laplacian(img, cv2.CV_64F).var()
        if uid not in var_dict.keys():
            var_dict[uid]={}
        if trackid not in var_dict[uid].keys():
            var_dict[uid][trackid]={}
        
        var_dict[uid][trackid][int(frameid)]=imageVar
        
    with open("train_var.json","w") as f:
        json.dump(var_dict,f)


args = argparser.parse_args()
train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path, args.gt_path,
                                   stride=args.train_stride, mode='train', transform=get_transform(False),args=args)
process(train_dataset.imgs,train_dataset.kframes,train_dataset.face_path)


with open("train_var.json","r") as f:
    var_dict = json.load(f)

var_array = [ ]
for (key1, uid_dict) in var_dict.items():
    for (key2, trackid_dict) in uid_dict.items():
        for (key3, value) in trackid_dict.items():
            var_array.append(value)

mean = np.mean(var_array)
std = np.sqrt(np.var(var_array, ddof=1))
p_dict={}
for (key1,uid_dict) in var_dict.items():
    if key1 not in p_dict.keys():
        p_dict[key1]={}
    for (key2,trackid_dict) in uid_dict.items():
        if key2 not in p_dict[key1].keys():
            p_dict[key1][key2]={}
        for(key3,value) in trackid_dict.items():
            p=norm.cdf(value,loc=mean,scale=std)
            p_dict[key1][key2][key3]=p

with open("train_p.json","w") as f:
    json.dump(p_dict,f)