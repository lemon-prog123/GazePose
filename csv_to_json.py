import json
import pandas as pd
import scipy.io as sio
import os

#path="3.json"
#file=open(path,"rb")
#fileJson = json.load(file)
#print(len(fileJson['results']))

records=[]
#for i in range(5):
#    print(f'pred{str(i)}.csv')
#    df=pd.read_csv(f'pred{str(i)}.csv',header=None)
#    d_records = df.to_dict('records')
#    records.append(d_records)

df=pd.read_csv('prednofilter.csv',header=None)
d_records = df.to_dict('records')
dic={"version": "1.0","challenge": "ego4d_looking_at_me"}
print(dic)
dic['results']=[]
leng = len(d_records)

for i, meta in enumerate(d_records):
    uid = meta[ 0 ]
    trackid = meta[ 2 ]
    unique_id = meta[ 1 ]
    #score=0
    #for j in range(5):
    #    score+=records[j][i][4]
    #    print(records[j][i][4])
    #score=score/5
    score = meta[ 4 ]
    label = 1
    new_dic = {'video_id': uid, 'unique_id': unique_id, 'track_id': trackid, 'label': label, 'score': score}
    dic[ 'results' ].append(new_dic)
    if i % 1000 == 0:
        print(i, leng)
print(len(dic['results']))

with open('158.json', 'w') as f:
    json.dump(dic, f)
#print(f)
#df=pd.read_csv('pred.csv',header=None)

#print(dic)#d_records = df.to_dict('records')
#print(d_records)

#dic={"version": "1.0","challenge": "ego4d_looking_at_me"}
