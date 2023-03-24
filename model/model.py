import torch, os, math, logging,sys
import torch.nn.functional as F
from model.transformer import Transformer
from model.position_encoding import PositionEmbeddingSine
from model.misc import NestedTensor
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.init import normal, constant
from model.resnet import resnet18,resnet50
#from model.transformer import make_encoder
from copy import deepcopy
logger = logging.getLogger(__name__)
from einops import rearrange


class GazePose(nn.Module):
    def __init__(self,args=None,input_size=15):
        super(GazePose, self).__init__()
        self.args=args
        self.img_feature_dim = 256
        if args.backbone=="resnet18":
            self.base_model = resnet18(pretrained=True,mesh=args.mesh)
            self.out_dim=512
        
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.body=self.args.body
        self.lstm=nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
        self.running_mean=torch.zeros(1)
        self.running_var=torch.ones(1)

        num_query=int(90/args.split)+1
        dmodel=args.dmodel
        print('dMoel')
        print(dmodel)

        print("Num Queries: "+str(num_query))
    
        num_tgt=0

        self.yaw_embed=nn.Embedding(num_query,dmodel)
        self.pitch_embed=nn.Embedding(num_query,dmodel)
        self.roll_embed=nn.Embedding(num_query,dmodel)
        num_tgt=3

        num_line=int(4/self.args.dsplit)+1
        num_net=num_line*num_line
        print("Num net "+str(num_net))
        self.joint_embed=nn.Embedding(num_net,dmodel//2)
        self.c_embed=nn.Embedding(int(1/args.csplit)+1,dmodel//2)
        num_tgt+=5

        print("Num Target: "+str(num_tgt))
        self.headpos_emeb=nn.Parameter(torch.randn(num_tgt,dmodel))
        self.transformer=Transformer(
                            d_model=dmodel,
                            dropout=0.1,
                            nhead=8,
                            dim_feedforward=args.dim_feedforward,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers,
                            normalize_before=True,
                            return_intermediate_dec=False,args=args)
        
        self.input_proj = nn.Conv2d(self.out_dim,dmodel, kernel_size=1)
        hidden_dim=dmodel
        N_steps = hidden_dim // 2
        self.position_embedding=PositionEmbeddingSine(N_steps, normalize=True)
        self.combine_layer=nn.Linear(4*self.img_feature_dim,self.img_feature_dim)
        self.fusion_layer=nn.Linear(num_tgt*dmodel, 2*self.img_feature_dim)
        self.last_layer=nn.Linear(self.img_feature_dim,2)
        self.cnt=0
        self.load_checkpoint()

    def _jointindex(self,head2D):
        length=5
        x=0
        y=5
        c=10
        num_line=int(2/self.args.dsplit)
        with torch.no_grad():
            for i in range(length):
                index=[i+x,i+y]
                xy_index=torch.floor((head2D[:,index]+2)/self.args.dsplit)
                xy_index[:,0]=xy_index[:,0]*num_line+xy_index[:,1]
                final_index=torch.LongTensor(xy_index[:,:].detach().cpu().numpy()).cuda()
                index=[i+c]
                c_index=torch.floor(head2D[:,index]/self.args.csplit)
                final_index[:,1]=c_index[:,0]
                if i==0:
                    return_index=final_index.unsqueeze(1)
                else:
                    final_index=final_index.unsqueeze(1)
                    return_index=torch.cat((return_index,final_index),1)

            tensor1=torch.cat((self.joint_embed(return_index[:,0,0]),self.c_embed(return_index[:,0,1])),1).unsqueeze(1)
            tensor2=torch.cat((self.joint_embed(return_index[:,1,0]),self.c_embed(return_index[:,1,1])),1).unsqueeze(1)
            tensor3=torch.cat((self.joint_embed(return_index[:,2,0]),self.c_embed(return_index[:,2,1])),1).unsqueeze(1)
            tensor4=torch.cat((self.joint_embed(return_index[:,3,0]),self.c_embed(return_index[:,3,1])),1).unsqueeze(1)
            tensor5=torch.cat((self.joint_embed(return_index[:,4,0]),self.c_embed(return_index[:,4,1])),1).unsqueeze(1)

            joint_query=torch.cat((tensor1,tensor2,tensor3,tensor4,tensor5),1)

        return joint_query
    
    def forward(self, input,query=None,head2D=None,max_index=None):
        B=input.shape[0]

        if max_index==None:
            max_index=3
            
        tgt=self.headpos_emeb.repeat((B,1,1)).cuda()
        yaw_query=self.yaw_embed(query[:,0]).unsqueeze(1)
        pitch_query=self.pitch_embed(query[:,1]).unsqueeze(1)
        roll_query=self.roll_embed(query[:,2]).unsqueeze(1)
        joint_query=self._jointindex(head2D)
        query=torch.cat((yaw_query,pitch_query,roll_query,joint_query),1)

            
        base_out,x=self.base_model(input.view((-1, 3) + input.size()[-2:]))
        x=x.view(B,7,self.out_dim,7,7)[:,3]
        b, c, h, w =x.shape
        mask = torch.zeros((b,224,224), dtype=torch.bool,device=x.device)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x= NestedTensor(x, mask)
        pos =(self.position_embedding(x).to(x.tensors.dtype)).cuda()
        src, mask = x.decompose()
        hs = self.transformer(self.input_proj(src),mask,query, pos,tgt=tgt)[0].squeeze()
        if len(hs.shape)==2:
            hs=hs.unsqueeze(1)
        hs=hs.flatten(1)
        output=self.fusion_layer(hs)

        base_out = base_out.view(B,7,self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3]   
        lstm_out=lstm_out.to(torch.float32)
        combine_out=torch.cat((lstm_out,output),1)
        output=self.combine_layer(combine_out)
        output = self.last_layer(output).view(-1,2)

        return output
    
    def load_checkpoint(self,retrain=False):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'Pure' in self.args.checkpoint:
                    state_dict=state
                else:
                    if 'module' in list(state["state_dict"].keys())[0]:
                        state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                    else:
                        state_dict = state["state_dict"]

                if ('gaze360_model.pth' in self.args.checkpoint) or (not self.args.val and not self.args.eval):
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                    print("Pop last layer")
                    
                print(state_dict.keys())
                self.load_state_dict(state_dict, strict=(self.args.val or self.args.eval))


class GazeLSTM(nn.Module):
    def __init__(self, args,input_size=15):
        super(GazeLSTM, self).__init__()
        self.args = deepcopy(args)
        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
        self.last_layer=nn.Linear(512,2)        
        self.load_checkpoint()

    
    def forward(self, input):
        batch_size=input.shape[0]
        input=rearrange(input, 'b t c h w -> (b t) c h w')
        base_out,_= self.base_model(input)
        base_out = base_out.view(batch_size,7,self.img_feature_dim)
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        lstm_out=lstm_out.to(torch.float32)
        output=self.last_layer(lstm_out)

        return output
                    
    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)         
                if 'module' in list(state["state_dict"].keys())[0]:
                    state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                else:
                    state_dict = state["state_dict"]
                if (('gaze360_model.pth' in self.args.checkpoint)) and not self.args.val and not self.args.eval:
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                    
                print(state_dict.keys())
                self.load_state_dict(state_dict, strict=self.args.eval)
        