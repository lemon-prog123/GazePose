import mediapipe as mp
import numpy as np
import cv2
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

print(os.getpid())


root_path="../videos_challenge"
cnt=0
jumpcnt=0
BG_COLOR = (192, 192, 192) # gray
mesh_dict={}

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.1) as pose:
  for root, dirs, files in os.walk(root_path):
        for file in files:
          if cnt%10000==0:
                print(cnt,jumpcnt)
          cnt+=1
          image = cv2.imread(os.path.join(root,file))
          try:
            image_height, image_width, _ = image.shape
          except:
            continue
          
          # Convert the BGR image to RGB before processing.
          results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
          if not results.pose_landmarks:
            continue
          root_split=root.split('/')
          trackid=root_split[-1]
          uid=root_split[-2]
          unique_id = file.split('_')[1].split('.')[0]
          if uid not in mesh_dict:
                mesh_dict[uid]={}
          if trackid not in mesh_dict[uid]:
                mesh_dict[uid][trackid]={}
          #frameid=int(file[5:10])
          mesh_dict[uid][trackid][unique_id]=[]
          lands=[0,2,5,9,10]
          for i in range(len(results.pose_landmarks.landmark)):
                if i not in lands:
                  continue
                x=results.pose_landmarks.landmark[i].x
                y=results.pose_landmarks.landmark[i].y
                z=results.pose_landmarks.landmark[i].z
                c=results.pose_landmarks.landmark[i].visibility
                mesh_dict[uid][trackid][unique_id].append((x,y,z,c))

import json
with open("mesh_"+"test.json","w") as f:
          json.dump(mesh_dict,f)