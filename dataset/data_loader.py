import sys, os
from operator import length_hint
import cv2, json, glob, logging
import torch
import hashlib
import torchvision.transforms as transforms
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict, OrderedDict
import json
import threading
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
logger = logging.getLogger(__name__)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def helper():
    return defaultdict(OrderedDict)

def pad_video(video):
    assert len(video) == 7
    pad_idx = np.all(video == 0, axis=(1, 2, 3))
    mid_idx = int(len(pad_idx) / 2)
    #mid_idx = index
    pad_idx[mid_idx] = False
    pad_frames = video[~pad_idx]
    pad_frames = np.pad(pad_frames, ((sum(pad_idx[:mid_idx]), 0), (0, 0), (0, 0), (0, 0)), mode='edge')
    pad_frames = np.pad(pad_frames, ((0, sum(pad_idx[mid_idx + 1:])), (0, 0), (0, 0), (0, 0)), mode='edge')
    return pad_frames.astype(np.uint8)


def check(track):
    inter_track = []
    framenum = []
    bboxes = []
    for frame in track:
        x = frame['x']
        y = frame['y']
        w = frame['width']
        h = frame['height']
        if (w <= 0 or h <= 0 or
                frame['frameNumber'] == 0 or
                len(frame['Person ID']) == 0):
            continue
        framenum.append(frame['frameNumber'])
        x = max(x, 0)
        y = max(y, 0)
        bbox = [x, y, x + w, y + h]
        bboxes.append(bbox)

    if len(framenum) == 0:
        return inter_track

    framenum = np.array(framenum)
    bboxes = np.array(bboxes)

    gt_frames = framenum[-1] - framenum[0] + 1

    frame_i = np.arange(framenum[0], framenum[-1] + 1)

    if gt_frames > framenum.shape[0]:
        bboxes_i = []
        for ij in range(0, 4):
            interpfn = interp1d(framenum, bboxes[:, ij])
            bboxes_i.append(interpfn(frame_i))
        bboxes_i = np.stack(bboxes_i, axis=1)
    else:
        frame_i = framenum
        bboxes_i = bboxes

    # assemble new tracklet
    template = track[0]
    for i, (frame, bbox) in enumerate(zip(frame_i, bboxes_i)):
        record = template.copy()
        record['frameNumber'] = frame
        record['x'] = bbox[0]
        record['y'] = bbox[1]
        record['width'] = bbox[2] - bbox[0]
        record['height'] = bbox[3] - bbox[1]
        inter_track.append(record)
    return inter_track


def make_dataset(file_name, json_path, gt_path, stride=1,args=None,dic=None,mode='train',head2D_dict=None,alphapose=False):
    logger.info('load: ' + file_name)
    # all videos
    cnt0=0
    cnt1=0

    images = []
    keyframes = []
    filter_frames=[]
    count = 0
    labels=[]
    lack_2D=0
    # video list
    with open(file_name, 'r') as f:
        videos = f.readlines()
        
    for uid in videos:
        uid = uid.strip()
        # per video
        # xxx.mp4.json
        with open(os.path.join(gt_path, uid + '.json')) as f:
            gts = json.load(f)
        positive = set()
        # load
        for gt in gts:
            for i in range(gt['start_frame'], gt['end_frame'] + 1):
                positive.add(str(i) + ":" + gt['label'])
        # json dir
        vid_json_dir = os.path.join(json_path, uid)
        # all faces
        tracklets = glob.glob(f'{vid_json_dir}/*.json')
        for t in tracklets:
            with open(t, 'r') as j:
                frames = json.load(j)
            frames.sort(key=lambda x: x['frameNumber'])
            trackid = os.path.basename(t)[:-5]
            # check the bbox, interpolate when necessary
            frames = check(frames)

            for idx, frame in enumerate(frames):
                frameid = frame['frameNumber']
                bbox = (frame['x'],
                        frame['y'],
                        frame['x'] + frame['width'],
                        frame['y'] + frame['height'])
                identifier = str(frameid) + ':' + frame['Person ID']
                label = 1 if identifier in positive else 0

                images.append((uid, trackid, frameid, bbox, label))
                key=uid+":"+trackid+":"+str(frameid)
                        
                if ((mode=='val' or mode=='train') and (idx% stride == 0)):
                    if dic!=None:
                        max_index=dic[key]['max_index']
                        max_score=dic[key]['array'][max_index]
                        if max_score<=0:
                            filter_frames.append(count)
                            count += 1
                            continue

                    if head2D_dict!=None:
                        try:
                            array=head2D_dict[uid][trackid][str(frameid)]
                            if alphapose:
                                array_dict={}
                                array_dict["keypoints"]=array
                                array_dict["bbox"]=(0,0,224,224)
                                array=array_dict
                            bbox=array['bbox']
                            w_bbox=bbox[2]-bbox[0]
                            h_bbox=bbox[3]-bbox[1]
                            if w_bbox==0 or h_bbox==0 or len(array['keypoints'])==0:
                                raise ValueError("can't find bbox")
                        except:
                            lack_2D+=1
                            
                    track_path=os.path.join('../social-interactions/data/face_imgs',uid,trackid)
                    face_img=f'{track_path}/face_{frameid:05d}.jpg'
                    
                    if not os.path.exists(face_img):
                        count+=1
                        continue
                    labels.append(label)
                    keyframes.append(count)
                    if images[count][4]==1:
                        cnt1+=1
                    else:
                        cnt0+=1
                count += 1
    
    print(cnt1,cnt0)
    print(f"{lack_2D} has been lacked")
    return images, keyframes,filter_frames,labels,(cnt1,cnt0)


def unified(point,x_mean,y_mean,w_bbox,h_bbox,w_image):
    x,y=point
    x_u=(x-x_mean)/w_bbox
    y_u=(y-y_mean)/h_bbox
    return x_u,y_u

def transformation(array,body=False,mesh=False):
    bbox=array['bbox']
    w_bbox=bbox[2]-bbox[0]
    h_bbox=bbox[3]-bbox[1]
    keypoints=array['keypoints']
    if mesh==True:
        alpha=224
    else:
        alpha=1
    
    right_eye=np.array(keypoints[1][:2])*alpha
    left_eye=np.array(keypoints[2][:2])*alpha

    mean_eye=(right_eye+left_eye)/2
    x_mean=mean_eye[0]
    y_mean=mean_eye[1]
    w_image=1080
    X=[]
    Y=[]
    S=[]
    for i,(x,y,score) in enumerate(keypoints):
        if i==5 and body==False:
            break

        x_u,y_u=unified((x*alpha,y*alpha),x_mean,y_mean,w_bbox,h_bbox,w_image)
        
        if x_u<-2:
            x_u=-2
        if x_u>2:
            x_u=2
        if y_u<-2:
            y_u=-2
        if y_u>2:
            y_u=2
        if score>1:
            score=1
        X.append(x_u)
        Y.append(y_u)
        S.append(score)

    kps_final_normalized=np.array([X, Y, S]).flatten().tolist()
    tensor_kps = torch.Tensor(kps_final_normalized)
    return tensor_kps


class ImagerLoader(torch.utils.data.Dataset):
    def __init__(self, source_path, file_name, json_path, gt_path,
                 stride=1, scale=0, mode='train', transform=None,args=None,test_dataset=None):

        self.file_name = file_name
        assert os.path.exists(self.file_name), f'{mode} list not exist'
        self.json_path = json_path
        assert os.path.exists(self.json_path), 'json path not exist'
        self.source_path=source_path
        self.face_path='../social-interactions/data/face_imgs'
        self.transform = transform

        if mode=="train":
            if args.mesh:
                with open(os.path.join(args.prior_path,"train_mesh.json"),'r') as f:
                    self.head2D_dict=json.load(f)
                print("Load Train Mesh")
            else:
                self.head2D_dict=None
        else:
            if args.mesh:
                with open(os.path.join(args.prior_path,"val_mesh.json"),'r') as f:
                    self.head2D_dict=json.load(f)
                print("Load Val Mesh")
            else:
                self.head2D_dict=None
        

        if mode=="train" and args.Gaussiandrop:
            with open(os.path.join(args.prior_path,'train_p.json'),"r") as f:
                self.p_dict=json.load(f)
        elif mode=="val" and args.Gaussiandrop:
            with open(os.path.join(args.prior_path,'val_p.json'),"r") as f:
                self.p_dict=json.load(f)

        if args.Gaussiandrop and mode=='train':
            file=open(os.path.join(args.prior_path,'train_filter_all.json'),'r')
            logger.info('Set Train Filter')
            self.dic=json.load(file)
            file.close()
        else:
            self.dic=None
            self.val_filter=None
        
        if mode=='train':
            file=open(os.path.join(args.prior_path,'train_headpose.json'),'r')
            logger.info('Load Train Headpose')
            self.dic3=None
            self.dic4=json.load(file)
            file.close()
        else:
            self.dic3=None
            file=open(os.path.join(args.prior_path,'val_headpose-all.json'),'r')
            logger.info('Load val Headpose')
            self.dic4=json.load(file)
            file.close()

        images, keyframes,filter_frames,labels,(lam,nlam)= make_dataset(file_name, json_path, gt_path, stride=stride,args=args,dic=self.dic,mode=mode,head2D_dict=self.head2D_dict,alphapose=args.mesh)
        self.lam=lam
        self.nlam=nlam
        self.args=args
        self.imgs = images
        self.kframes = keyframes
        self.olen=len(self.kframes)
        self.labels=labels
        self.filter_frames=filter_frames
        self.img_group = self._get_img_group()
        self.scale = scale  # box expand ratio
        self.mode = mode

    def get_labels(self):
        return self.labels

    def get_index(self,angle):
        return angle/self.args.split

    def __getitem__(self, index):
        source_video,max_index = self._get_video(index)
        target = self._get_target(index)

        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        key=uid+":"+trackid+":"+str(frameid)
        try:
            yaw,pitch,roll,_,_,_=self.dic4[key]
        except:
            yaw=pitch=roll=90

        yaw=np.abs(yaw)
        pitch=np.abs(pitch)
        roll=np.abs(roll)

        yaw=self.get_index(yaw)
        pitch=self.get_index(pitch)
        roll=self.get_index(roll)
        angle=np.floor(np.array([yaw,pitch,roll]))
        angle=torch.LongTensor(angle)

        if self.args.mesh:
            head2D_flag=False# Not Use Filter
            dis_array=[0,-1,1,-2,2,3,-3]
            uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]

            for dis in dis_array:
                i=frameid+dis
                try:
                    array=self.head2D_dict[uid][trackid][str(i)]
                except:
                    continue
                if self.args.mesh:
                    array_dict={}
                    if self.args.mesh:
                        array_dict["keypoints"]=np.array(array)[:,[0,1,3]]
                    else:
                        array_dict["keypoints"]=array
                    array_dict["bbox"]=(0,0,224,224)
                    array=array_dict
                
                bbox=array['bbox']
                w_bbox=bbox[2]-bbox[0]
                h_bbox=bbox[3]-bbox[1]
                if w_bbox==0 or h_bbox==0 or len(array['keypoints'])==0:
                    raise ValueError("can't find bbox")
                head2D_flag=True
                head2D=transformation(array,body=self.args.body,mesh=self.args.mesh)
                break
            
            if head2D_flag==False:
                head2D=torch.zeros([15],dtype=torch.float)
        else:
            head2D=torch.zeros([15],dtype=torch.float)

        return source_video, target,angle,head2D,max_index

    def _get_video(self, index, debug=False):
        flag=False
        import random
        if random.random()>0.5 and self.mode=="train":
            flag=True
        
        uid, trackid, frameid, _, label = self.imgs[self.kframes[index]]
        video = []
        need_pad = False
        key=uid+":"+trackid+":"+str(frameid)  

        if self.dic!=None:  
            max_index=self.dic[key]['max_index']
        elif self.val_filter!=None:
            max_index=self.val_filter[key]['max_index']
        else:
            max_index=3
    
        for i in range(frameid - 3, frameid + 4):
            uid_path=os.path.join(self.face_path,uid)
            track_path=os.path.join(uid_path,trackid)
            face_img=f'{track_path}/face_{i:05d}.jpg'
            Gaussian_flag=False

            if self.args.Gaussiandrop and i!=frameid and self.mode=="train":
                try:
                    p=self.p_dict[uid][trackid][str(i)]
                except:
                    p=1
                if random.random() > p+0.2:
                    Gaussian_flag=True
            elif self.args.Gaussiandrop and i!=frameid and self.mode=="val":
                try:
                    p=self.p_dict[uid][trackid][str(i)]
                except:
                    p=1
                if p<0.2:
                    Gaussian_flag=True

            if i not in self.img_group[uid][trackid] or not os.path.exists(face_img) or Gaussian_flag==True:
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
                continue

            assert os.path.exists(face_img), f'img: {face_img} not found'
            img = cv2.imread(face_img)
            if flag==True:
                img = cv2.flip(img,1)
            face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(face)
                plt.show()
            video.append(np.expand_dims(face, axis=0))

        video = np.concatenate(video, axis=0)
        
        if need_pad:
            video = pad_video(video)
            
        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32),max_index
    
    def _get_target(self, index):
        if index>=self.olen:
            index=index-self.olen
        
        if self.mode == 'train':
            return self.imgs[self.kframes[index]]
        else:
            return self.imgs[self.kframes[index]]

    def _get_img_group(self):
        img_group = self._nested_dict()
        for db in self.imgs:
            img_group[db[0]][db[1]][db[2]] = db[3]
        return img_group

    def _nested_dict(self):
        return defaultdict(helper)
    
    def __len__(self):
        return len(self.kframes)


def make_test_dataset(test_path, stride=1,face_dict=None):
    logger.info('load: ' + test_path)
    g = os.walk(test_path)
    images = []
    keyframes = []
    count = 0
    cnt1=0
    cnt2=0
    lack=0
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            if os.path.exists(os.path.join(test_path, dir_name)):
                uid = dir_name
                g2 = os.walk(os.path.join(test_path, uid))
                for _, track_list, _ in g2:
                    for track_id in track_list:
                        g3 = os.walk(os.path.join(test_path, uid, track_id))
                        for _, _, frame_list in g3:
                            for idx, frames in enumerate(frame_list):
                                frame_id = frames.split('_')[0]
                                unique_id = frames.split('_')[1].split('.')[0]
                                images.append((uid, track_id, unique_id, frame_id))
                                if idx % stride == 0:
                                    if face_dict!=None:
                                        try:
                                            array=face_dict[uid][track_id][unique_id]
                                            array_dict={}
                                            array_dict["keypoints"]=array
                                            array_dict["bbox"]=(0,0,224,224)
                                            array=array_dict
                                            bbox=array['bbox']
                                            w_bbox=bbox[2]-bbox[0]
                                            h_bbox=bbox[3]-bbox[1]
                                            if w_bbox==0 or h_bbox==0 or len(array['keypoints'])==0:
                                                raise ValueError("can't find bbox")
                                        except:
                                            lack+=1
                                    keyframes.append(count)
                                count += 1
    print(cnt1,cnt2)
    print(str(lack)+" has been lacked")
    return images, keyframes

class TestImagerLoader(torch.utils.data.Dataset):
    def __init__(self, test_path,args,stride=1, transform=None):

        self.test_path = test_path
        assert os.path.exists(self.test_path), 'test dataset path not exist'


        self.args=args
        if self.args.mesh:
            with open(os.path.join(args.prior_path,"test_mesh.json"),"r") as f:
                self.head2D_dict=json.load(f)
        else:
            self.head2D_dict=None
        images, keyframes = make_test_dataset(test_path, stride=stride,face_dict=self.head2D_dict)
        self.imgs = images
        self.kframes = keyframes
        self.transform = transform
        self.args = args

        file=open(os.path.join(args.prior_path,'test_headpose.json'),'r')
        logger.info('Load Test Headpose')
        self.dic=json.load(file)
        file.close()

        if args.Gaussiandrop:
            with open(os.path.join(args.prior_path,'test_p.json'),"r") as f:
                self.p_dict=json.load(f)

    def get_index(self,angle):
        return angle/self.args.split

    def __getitem__(self, index):
        source_video,max_index = self._get_video(index)
        target = self._get_target(index)

        uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
        key=uid+":"+trackid+":"+uniqueid+":"+str(frameid)
        yaw,pitch,roll=self.dic[key]
        yaw=np.abs(yaw)
        pitch=np.abs(pitch)
        roll=np.abs(roll)

        yaw=self.get_index(yaw)
        pitch=self.get_index(pitch)
        roll=self.get_index(roll)
        angle=np.floor(np.array([yaw,pitch,roll]))
        angle=torch.LongTensor(angle)

        head2D_flag=False
        if self.args.mesh:
            uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
            try:
                array=self.head2D_dict[uid][trackid][uniqueid]
                array_dict={}
                if self.args.mesh:
                    array_dict["keypoints"]=np.array(array)[:,[0,1,3]]
                else:
                    array_dict["keypoints"]=array
                array_dict["bbox"]=(0,0,224,224)
                array=array_dict
                        
                bbox=array['bbox']
                w_bbox=bbox[2]-bbox[0]
                h_bbox=bbox[3]-bbox[1]
                if w_bbox==0 or h_bbox==0 or len(array['keypoints'])==0:
                    raise ValueError("can't find bbox")
                head2D_flag=True
                head2D=transformation(array,mesh=self.args.mesh)
            except:
                pass
            if head2D_flag==False:
                head2D=torch.zeros([15],dtype=torch.float)
        else:
            head2D=torch.zeros([15],dtype=torch.float)
        
        return source_video, target,angle,head2D,max_index
    
    def _get_video(self, index):
        uid, trackid, uniqueid, frameid = self.imgs[self.kframes[index]]
        video = []
        need_pad = False
        key=uid+":"+trackid+":"+uniqueid+":"+str(frameid)
        max_index=3

        path = os.path.join(self.test_path, uid, trackid)
        cnt=0
        for i in range(int(frameid) - 3, int(frameid) + 4):
            found = False
            ii = str(i).zfill(5)
            g = os.walk(path)
            Gaussian_flag=False
            
            for _, _, file_list in g:
                for f in file_list:
                    if ii in f:
                        frame_id = f.split('_')[0]
                        unique_id = f.split('_')[1].split('.')[0]
                        key=uid+":"+trackid+":"+unique_id+":"+str(frame_id)
                        if self.args.Gaussiandrop and i!=frameid:
                            p=self.p_dict[uid][trackid][unique_id]
                            if p<0.2:
                                Gaussian_flag=True
                                break
                        img_path = os.path.join(path, f)
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        video.append(np.expand_dims(img, axis=0))
                        found = True
                        break
            if not found or Gaussian_flag:
                video.append(np.zeros((1, 224, 224, 3), dtype=np.uint8))
                if not need_pad:
                    need_pad = True
            cnt+=1

        video = np.concatenate(video, axis=0)
        if need_pad:
            video = pad_video(video)

        if self.transform:
            video = torch.cat([self.transform(f).unsqueeze(0) for f in video], dim=0)

        return video.type(torch.float32),max_index
    def __len__(self):
        return len(self.kframes)
    
    def _get_target(self, index):
        return self.imgs[self.kframes[index]]
