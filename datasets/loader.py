import os
import os.path as osp
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, random_split
import core.distributed as dist

from core.config import cfg
from datasets.google_landmark_v2 import GoogleLandMark

def _construct_loader(data_path, mode, 
                      batch_size, shuffle, 
                      drop_last, num_workers,
                      pin_mem):
    dataset = GoogleLandMark(data_path, mode)
    sampler = DistributedSampler(dataset) if dist.get_world_size() > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=(False if sampler else shuffle), 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last)

    return loader


def _construct_train_loader():
    return _construct_loader(
        data_path=cfg.TRAIN.DATASET_ROOT,
        mode=cfg.TRAIN.MODE, 
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.NUM_GPUS),
        shuffle=True,
        drop_last=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_mem=cfg.TRAIN.PIN_MEMORY
    )


def _construct_test_loader():
    return _construct_loader(
        data_path=cfg.TEST.DATASET_ROOT,
        mode=cfg.TEST.MODE,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.OPTIM.NUM_GPUS),
        shuffle=False,
        drop_last=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_mem=cfg.TEST.PIN_MEMORY
    )