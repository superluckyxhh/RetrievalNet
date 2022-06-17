from cProfile import label
import os
import torch
import numpy as np
import argparse
import json

import core.optim as optim
import core.distributed as dist
import core.net as net

from core.utils import reduce_dict, scaled_all_reduce
from core.logger import Logger, SmoothedValue, MetricLogger

from datasets.loader import _construct_train_loader, _construct_test_loader
from core.loss import topk_errors, setup_criterion, setup_global_criterion, set_local_criterion
from core.config import cfg
from models.retrievalnet import RetrievalNet

# DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def setup_env(args):
    dist.init_dist_mode(args)
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Seed Used: ', seed)


def setup_model():
    model = RetrievalNet(backbone=cfg.MODEL.BACKBONE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params:', n_params)
    
    cur_dev = torch.cuda.current_device()
    model = model.cuda(device=cur_dev)
    
    return model, n_params


def train_epoch(train_loader, model, optimizer, criterion, cur_epoch, print_freq=1, tb_logger=None):
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_epoch_lr(optimizer, lr)
    
    model.train()
    criterion.train()
    
    logger = MetricLogger(delimiter='  ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{cur_epoch}]'
    
    for ims, labels in logger.log_every(train_loader, print_freq, header):
        ims = ims.cuda()
        labels = labels.cuda(non_blocking=True)
        
        global_feats = model(ims)  
        
        # Global criterion
        loss_dict = criterion(global_feats, labels)
        # Local criterion
        # loss_dict = criterion(local_feats, labels)

        # Total Loss
        # loss_dict = criterion(global_feats, local_feats, labels)

        loss = loss_dict['loss']
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_item = {
            k: v.item() for k, v in loss_dict_reduced.items()
        }
        
        logger.update(**loss_dict_reduced_item)
        logger.update(lr=lr)
        if tb_logger is not None and dist.is_main_process():
            tb_logger.add_scalars(loss_dict_reduced, prefix='train')
            
    logger.synchronize_between_processes()
    print('Average train stats:', logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}

                
@torch.no_grad()
def test_epoch(test_loader, model, cur_epoch, print_freq=1, tb_logger=None):
    model.eval()
    
    logger = MetricLogger(delimiter='  ')
    header = f'Epoch: [{cur_epoch}]'
    
    for ims, labels in logger.log_every(test_loader, print_freq, header):
        ims = ims.cuda()
        labels = labels.cuda(non_blocking=True)
        
        global_feats = model(ims)  
        
        top1_err, top5_err = topk_errors(global_feats, labels, [1, 5])
        top1_err, top5_err = scaled_all_reduce([top1_err, top5_err])
        top1_err, top5_err = top1_err.item(), top5_err.item()
        
        topk_errs = {'top1_err': top1_err, 'top5_err': top5_err}
        logger.update(**topk_errs)
        
        if tb_logger is not None and dist.is_main_process():
            tb_logger.add_scalars(topk_errs, prefix='test')
    
    logger.synchronize_between_processes()
    print('Average test stats:', logger)
    return {k: meter.global_avg for k, meter in logger.meters.items()}

        

def train_model(args):
    setup_env(args)
    
    train_loader = _construct_train_loader()
    test_loader = _construct_test_loader()
    
    model, num_params = setup_model()
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[args.gpu],
                find_unused_parameters=True)
        model_without_ddp = model.module
    
    if cfg.MODEL.LOAD is not None:
        state_dict = torch.load(cfg.MODEL.LOAD)
        model.load_state_dict(state_dict['model'])
        print('Load Model Weights Done')
        
    # criterion = setup_criterion()
    # criterion = set_local_criterion()
    criterion = setup_global_criterion()
    criterion = criterion.cuda()
    optimizer = optim.construct_optimizer(model)
    
    artifact_name = (
        f'0617_{cfg.MODEL.NAME}_backbone_{cfg.MODEL.BACKBONE}_' +
        f'batch_{cfg.TRAIN.BATCH_SIZE}_' + 
        f'lr_{cfg.OPTIM.BASE_LR}_wamup_{cfg.OPTIM.WARMUP_EPOCHS}_' +
        f'globaloss_{cfg.LOSS.GLOBAL_NAME}'
    )
    artifact_path = os.path.join(cfg.MODEL.SAVEPATH, artifact_name)
    os.makedirs(artifact_path, exist_ok=True)
    
    tb_logger = Logger(artifact_path) if dist.is_main_process() else None
    
    for cur_epoch in range(cfg.OPTIM.MAX_EPOCHS):
        train_stats = train_epoch(
            train_loader=train_loader,
            model=model, 
            optimizer=optimizer, 
            criterion=criterion, 
            cur_epoch=cur_epoch, 
            print_freq=cfg.LOG.PRINT_FREQ,
            tb_logger=tb_logger
        )
        
        # Save the model
        if cur_epoch % cfg.LOG.SAVE_INTERVAL == 0 or cur_epoch == cfg.OPTIM.MAX_EPOCHS - 1:
            if dist.is_main_process():
                torch.save({'model': model_without_ddp.state_dict()},
                           f'{artifact_path}/model-epoch{cur_epoch}.pth')
        train_log_state = {
            'epoch': cur_epoch,
            'num_params': num_params,
            **{f'train_{k}': v for k, v in train_stats.items()}
        }
        with open(f'{artifact_path}/train.log', 'a') as f:
            f.write(json.dumps(train_log_state) + '\n')
            
        # Evaluate the model
        if cur_epoch % cfg.LOG.EVAL_PERIOD == 0 or cur_epoch == cfg.OPTIM.MAX_EPOCHS - 1:
            test_stats = test_epoch(test_loader, model, cur_epoch, print_freq=cfg.LOG.TEST_PRINT_FREQ, tb_logger=tb_logger)
        
        test_log_state = {
            'epoch': cur_epoch,
            'num_params': num_params,
            **{f'test{k}': v for k, v in test_stats.items()}
        }
        with open(f'{artifact_path}/test.log', 'a') as f:
            f.write(json.dumps(test_log_state) + '\n')
            
    print('Train Finished')
    
        
def test_model(args, model_weights):
    setup_env(args)
    model, num_params = setup_model()
    
    if model_weights is not None:
        state_dict = torch.load(model_weights)
        model.load_state_dict(state_dict['model'])        
        print('Load Model Weights Done')
        
    cur_epoch = 1
    test_loader = _construct_test_loader()
    test_epoch(test_loader, model, cur_epoch, print_freq=cfg.LOG.TEST_PRINT_FREQ, tb_logger=None)