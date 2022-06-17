import cv2
import os
import os.path as osp
from logzero import logger
from torch.utils.data import Dataset, DataLoader

from core.config import cfg
from datasets.transform_v2 import train_transform, test_transform


class GoogleLandMark(Dataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.imdb = []
        self.labels = []
        if mode == 'train':
            train_csv = osp.join(root, cfg.TRAIN.DATASET_FILE) 
            self._construct(train_csv)
            
        elif mode == 'test':
            test_csv = osp.join(root, cfg.TEST.DATASET_FILE)
            self._construct(test_csv)
       
    def _construct(self, path):
        with open(path, 'r') as f:
            raw = f.readlines()
        for i in range(len(raw)):
            im_name, im_label = raw[i].strip('\n').split(' ')
            im_path = osp.join(self.root, im_name)
            im_label = int(im_label)
            
            self.imdb.append({"image_path":im_path, "image_label":im_label})
            self.labels.append(im_label)
            
        im_info = f'The number of {self.mode} images is {len(self.imdb)}'
        label_info = f'The number of {self.mode} labels is {len(set(self.labels))}'
        logger.info(im_info)  
        logger.info(label_info)
    
    def _process(self, im):
        if self.mode == 'train':
            transformed = train_transform(image=im)
            im = transformed["image"] 
        else:
            transformed = test_transform(image=im)
            im = transformed["image"]
        
        return im

    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):
        try:
            im_path = self.imdb[index]['image_path']
            im = cv2.imread(im_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        except:
            raise ValueError('image path {} is not found'.format(self.imdb[index]['image_path']))
        
        im = self._process(im)
        label = int(self.imdb[index]['image_label'])
    
        return im, label