import os, torch, glob, subprocess, shutil
import torch.nn as nn
from torch import Tensor
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
from common.metrics import run_evaluation
from math import cos, pi
from albumentations import (Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90, RandomBrightness,RandomContrast,GaussianBlur,MotionBlur,RandomScale,RandomCropFromBorders,CenterCrop,ColorJitter,Rotate)
import torch.nn.functional as F


def get_transform(is_train):
    if is_train:
        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.RandomResizedCrop(size=224,scale=(0.8,1)),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])          
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])])
        
    return transform

def summary(exp_path,n):
    merge_path = f'{exp_path}/result'
    preds=[]
    gt=[]
    for i in range(n):
        tag=str(i)
        pred_file = f'{merge_path}/pred{tag}.csv'
        df=pd.read_csv(pred_file,header=None)
        d_records = df.to_dict('records')
        preds.append(d_records)
    
    gt=pd.read_csv(f'{merge_path}/gt0.csv',header=None)
    length=len(preds[0])
    for idx in range(length):
        final_score=0
        for i in range(n):
            #print(preds[i][idx])
            final_score+=preds[i][idx][8]
        final_score=final_score/n
        preds[0][idx][8]=final_score
    pred=[]
    for idx in range(length):
        meta=[]
        for j in range(9):
            meta.append(preds[0][idx][j])
        pred.append(meta)
    #print(pred[0])
    pred_file = f'{merge_path}/pred.csv'
    gt_file=f'{merge_path}/gt.csv'
    pred=pd.DataFrame(pred)
    pred.to_csv(pred_file, index=False, header=None)
    gt.to_csv(gt_file, index=False, header=None)
    run_evaluation(gt_file, pred_file)


class PostProcessor():
    def __init__(self, args,mode='val',tag=None):
        if mode=='val':
            self.exp_path = args.exp_path
        elif mode=='gaze':
            self.exp_path = args.train_path+'gaze'
        elif mode=='head':
            self.exp_path = args.train_path+'head'
        else:
            self.exp_path = args.train_path

        self.save_path = f'{self.exp_path}/tmp'
        self.tag=tag
        if not os.path.exists(self.exp_path):
            os.mkdir(self.exp_path)
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            
        self.groundtruth = []
        self.prediction = []
        if tag!=None:
            self.groundtruthfile = f'{self.save_path}/gt{tag}.csv.rank.{args.rank}'
            self.predctionfile = f'{self.save_path}/pred{tag}.csv.rank.{args.rank}'
        else:
            self.groundtruthfile = f'{self.save_path}/gt.csv.rank.{args.rank}'
            self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs, dim=-1)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            frameid = targets[2][idx].item()
            x1 = targets[3][0][idx].item()
            y1 = targets[3][1][idx].item()
            x2 = targets[3][2][idx].item()
            y2 = targets[3][3][idx].item()
            label = targets[4][idx].item()
            self.groundtruth.append([uid, frameid, x1, y1, x2, y2, trackid, label])
            self.prediction.append([uid, frameid, x1, y1, x2, y2, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.groundtruthfile):
            os.remove(self.groundtruthfile)
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        gt_df = pd.DataFrame(self.groundtruth)
        gt_df.to_csv(self.groundtruthfile, index=False, header=None)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)
    
    def get_mAP(self):
        # merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)
        
        if self.tag!=None:
            gt_file = f'{merge_path}/gt{self.tag}.csv'
        else:
            gt_file = f'{merge_path}/gt.csv'
        
        if os.path.exists(gt_file):
            os.remove(gt_file)
        if self.tag!=None:
            gts = glob.glob(f'{self.save_path}/gt{self.tag}.csv.rank.*')
        else:
            gts = glob.glob(f'{self.save_path}/gt.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(gts), gt_file)
        subprocess.call(cmd, shell=True)

        if self.tag!=None:
            pred_file = f'{merge_path}/pred{self.tag}.csv'
        else:
            pred_file = f'{merge_path}/pred.csv'
        if os.path.exists(pred_file):
            os.remove(pred_file)
        
        if self.tag!=None:
            preds = glob.glob(f'{self.save_path}/pred{self.tag}.csv.rank.*')
        else:
            preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)
        return run_evaluation(gt_file, pred_file)


class TestPostProcessor():
    def __init__(self, args):
        self.exp_path = args.exp_path
        self.save_path = f'{self.exp_path}/tmp'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.prediction = []
        self.predctionfile = f'{self.save_path}/pred.csv.rank.{args.rank}'

    def update(self, outputs, targets):
        # postprocess outputs of one minibatch
        outputs = F.softmax(outputs, dim=-1)
        for idx, scores in enumerate(outputs):
            uid = targets[0][idx]
            trackid = targets[1][idx]
            unique_id = targets[2][idx]
            self.prediction.append([uid, unique_id, trackid, 1, scores[1].item()])

    def save(self):
        if os.path.exists(self.predctionfile):
            os.remove(self.predctionfile)
        pred_df = pd.DataFrame(self.prediction)
        pred_df.to_csv(self.predctionfile, index=False, header=None)

        #merge csv
        merge_path = f'{self.exp_path}/result'
        if not os.path.exists(merge_path):
            os.mkdir(merge_path)
        pred_file = f'{merge_path}/pred.csv'
        preds = glob.glob(f'{self.save_path}/pred.csv.rank.*')
        cmd = 'cat {} > {}'.format(' '.join(preds), pred_file)
        subprocess.call(cmd, shell=True)
        shutil.rmtree(self.save_path)


def save_checkpoint(state, save_path, is_best=False, is_dist=False):
    save_path = f'{save_path}/checkpoint'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    epoch = state['epoch']
    filename = f'{save_path}/epoch_{epoch}.pth'
    torch.save(state, filename)

    if is_best:
        if os.path.exists(f'{save_path}/best.pth'):
            os.remove(f'{save_path}/best.pth')
        torch.save(state, f'{save_path}/best.pth')


def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count