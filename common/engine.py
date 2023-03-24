import os, logging,sys
from re import A, S
import time
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from common.utils import AverageMeter
from common.utils import save_checkpoint, PostProcessor
import json
from einops import rearrange
logger = logging.getLogger(__name__)


def train(train_loader, model, criterion, optimizer, epoch,postprocess,args,lr_scheduler=None,steps=0,num_steps=0,val_params=None,best_mAP=0,global_step=1):
    logger.info('training')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    avg_loss = AverageMeter()
    model.train()

    end = time.time()

    for i,  (source_frame, target,angle,head2D,max_index) in enumerate(train_loader):
        steps+=1
        # measure data loading time
        data_time.update(time.time() - end)

        max_index=torch.LongTensor(max_index).cuda()
        head2D=head2D.cuda()
        source_frame=source_frame.cuda()
        label = target[4]
        label=torch.LongTensor(label).cuda()
        angle=angle.cuda()
        
        if args.mesh:
            output=model(source_frame,query=angle,head2D=head2D,max_index=max_index)
        else:
            output=model(source_frame)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss.update(loss.item())
        postprocess.update(output.detach().cpu(), target)

        batch_time.update(time.time() - end)
        end = time.time()


        if i % 100 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Step {Step:d}\t'
                        'Lr  {Lr:.8f}\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=avg_loss,Step=steps,Lr=optimizer.param_groups[0]['lr']))
            
        if i%4000==0 and i!=0:
            postprocess_val = PostProcessor(args)
            val_params['postprocess']=postprocess_val
            with torch.no_grad():
                mAP,loss = validate(**val_params)
            model.train()

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
            if is_best:
                save_checkpoint({
                    'epoch': global_step,
                    'state_dict': model.state_dict(),
                    'mAP': mAP},
                    save_path=args.exp_path,
                    is_best=is_best,
                    is_dist=args.dist)
    
    postprocess.save()
    mAP=0
    mAP = postprocess.get_mAP()

    return avg_loss.avg,mAP,steps,best_mAP,global_step


def validate(val_loader,model, postprocess,args,mode='val',weight=None):
    logger.info('evaluating')
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    model.eval()
    end = time.time()

    for i, (source_frame, target,angle,head2D,max_index) in enumerate(val_loader):
        max_index=torch.LongTensor(max_index).cuda()
        head2D=head2D.cuda()
        source_frame=source_frame.cuda()
        angle=angle.cuda()

        if not mode=='test':
            targets=target[4].cuda()

        with torch.no_grad():

            if args.mesh:
                output=model(source_frame,query=angle,head2D=head2D,max_index=max_index)
            else:
                output=model(source_frame)

            if not mode=='test':
                loss=F.cross_entropy(output,targets,weight=weight)
                avg_loss.update(loss.item())
            postprocess.update(output.detach().cpu(), target)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                logger.info('Processed: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.avg:.3f}\t'.format(
                        i, len(val_loader), batch_time=batch_time,loss=avg_loss))
    
    postprocess.save()

    if mode == 'val':
        mAP = None
        mAP = postprocess.get_mAP()
        return mAP,avg_loss.avg
    elif mode == 'test':
        print('generate pred.csv')