from common.config import argparser
import os, sys, random, pprint
import torch
from common.logger import create_logger
from dataset.data_loader import  ImagerLoader, TestImagerLoader
from common.utils import PostProcessor, get_transform, save_checkpoint, TestPostProcessor
from common.engine import train, validate
from model.model import GazePose,GazeLSTM

def main(args):
    print('PID')
    print(os.getpid())
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        torch.backends.cudnn.enabled = True
        torch.cuda.init()

    if  not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)
    m_seed = 3407

    torch.manual_seed(m_seed)
    torch.cuda.manual_seed_all(m_seed)

    logger = create_logger(args)
    logger.info(pprint.pformat(args))
    logger.info(f'Model: {args.model}')

    model = eval(args.model)(args=args,input_size=15)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    params = {'shuffle': True}
    if not args.eval:
        if not args.val:
            train_dataset = ImagerLoader(args.source_path, args.train_file, args.json_path,
                                            args.gt_path, stride=args.train_stride, transform=get_transform(True),args=args)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                **params)
        val_dataset = ImagerLoader(args.source_path, args.val_file, args.json_path, args.gt_path,
                                    stride=args.val_stride, mode='val', transform=get_transform(False),args=args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            **params)
    else:
        test_dataset = TestImagerLoader(args.test_path,args,stride=args.test_stride, transform=get_transform(False))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            **params)
    if not args.val and not args.eval:
        lam=train_dataset.lam
        nlam=train_dataset.nlam
        nlam=lam/(lam+nlam)
        lam=1-nlam
        class_weights=torch.FloatTensor([nlam,lam]).cuda()
    else:
        class_weights = torch.FloatTensor(args.weights).cuda()

    print(class_weights)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print('Use all module to train')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_mAP = 0

    if not args.val and not args.eval:
        num_steps = len(train_loader) * args.epochs
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=(5e-6))


    train_loss=[]
    val_loss=[]
    val_map=[]
    train_map=[]
    global_step=1
    if not args.eval and not args.val:
        logger.info('start training')
        steps=0
        postprocess_train = PostProcessor(args,mode='train')
        val_params={'val_loader':val_loader,'model':model,'postprocess':None,'args':args,
            'mode':'val','weight':class_weights}
        
        for epoch in range(args.epochs):
            loss,mAP,steps,mAP_epoch,global_step=train(train_loader, model, criterion, optimizer, epoch,postprocess_train,
                                                    args,CosineLR,steps,num_steps,val_params=val_params,best_mAP=best_mAP,global_step=global_step)
            best_mAP = max(best_mAP, mAP_epoch)
            train_loss.append(loss)
            train_map.append(mAP)
            logger.info(f'mAP: {mAP:.4f}')

            postprocess_val = PostProcessor(args)
            mAP,loss = validate(val_loader,model, postprocess_val, args=args,mode='val',weight=class_weights)
            val_map.append(mAP)
            val_loss.append(loss)

            global_step+=1
            print('Global Step:'+str(global_step))
            print(train_loss)
            print(train_map)
            print(val_map)
            print(val_loss)
            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
            save_checkpoint({
                'epoch': global_step,
                'state_dict': model.state_dict(),
                'mAP': mAP},
                save_path=args.exp_path,
                is_best=is_best,
                is_dist=args.dist)
            CosineLR.step()
    elif args.val:
        postprocess = PostProcessor(args,tag=None)
        mAP,loss = validate(val_loader,model, postprocess, args=args,mode='val',weight=class_weights)
        val_map.append(mAP)
        val_loss.append(loss)

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        if args.model=="BaselineLSTM":
            logger.info(f'mAP: {mAP:.4f} best mAP: {best_mAP:.4f}')
            logger.info(f'mean: {model.running_mean.item():.4f} var: {model.running_var.item():.4f}')
    else:
        logger.info('start evaluating')
        postprocess = TestPostProcessor(args)
        validate(test_loader,model, postprocess, mode='test',args=args,weight=class_weights)
    
def run():
    args = argparser.parse_args()
    main(args)
if __name__ == '__main__':
    run()
