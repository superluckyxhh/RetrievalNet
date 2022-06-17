import cv2
import os
import torch
import numpy as np
from scipy.io import savemat
from PIL import Image, ImageFile

from dataset import configdataset
from download import download_datasets
from models.retrievalnet import RetrievalNet

""" Online Extract """

data_root = '/home/user/dataset/data'
save_root = '/home/user/code/RetrievalNet/revisitop'
save_path = os.path.join(save_root, test_dataset + '.mat')

weight_path = None

# Set test dataset: roxford5k | rparis6k
test_dataset = 'roxford5k'
# test_dataset = 'rparis6k'

def pil_loader(path):
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning 
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

print('>> {}: Processing test dataset...'.format(test_dataset)) 
# config file for the dataset
# separates query image list from database image list, if revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

model = RetrievalNet('resnet101', 81313)

if weight_path is not None:
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    print('Load Model Weights Done')
else:
    print('Model Weight Is Not Found')

# query images
q_descs = []
for i in np.arange(cfg['nq']):
    im_path = cfg['qim_fname'](cfg, i)
    qim = pil_loader(im_path).crop(cfg['gnd'][i]['bbx'])
    np_qim = np.array(qim)
    tensor_qim = torch.tensor(np_qim, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()

    with torch.no_grad():
        global_desc = model(tensor_qim)
    q_descs.append(global_desc)  
    print('>> {}: Processing query image {}'.format(test_dataset, i+1))

q_descs = torch.cat(q_descs, dim=0).transpose(0, 1)
q_descs_np = q_descs.detach().cpu().numpy()

im_descs = []
for i in np.arange(cfg['n']):
    if i == 20:
        break
    im_path = cfg['im_fname'](cfg, i)
    im = pil_loader(im_path)
    np_im = np.array(im)
    tensor_im = torch.tensor(np_im, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    
    with torch.no_grad():
        global_desc = model(tensor_im)
    im_descs.append(global_desc)  
    print('>> {}: Processing database image {}'.format(test_dataset, i+1))

im_descs = torch.cat(im_descs, dim=0).transpose(0, 1)
im_descs_np = im_descs.detach().cpu().numpy()

savemat(save_path, {"X": im_descs_np, "Q": q_descs_np})
print(f'{test_dataset} Query and Database Features Save Done')