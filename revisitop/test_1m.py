import argparse
import fnmatch
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models.retrievalnet import RetrievalNet

from revisitop.dataset import configdataset
from revisitop.download import download_datasets, download_distractors
from revisitop.genericdataset import ImagesFromList
from revisitop.evaluate import compute_map_and_print
from revisitop.utils import htime



datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k', 'revisitop1m']

# test options
parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Example')
parser.add_argument('--weights', '-n', metavar='WEIGHTS', default='', 
                    help="network to be evaluated. " )
parser.add_argument('--datasets', '-d', metavar='DATASETS', default='revisitop1m',
                    help="comma separated list of test datasets: " +
                        " | ".join(datasets_names) +
                        " (default: 'roxford5k,rparis6k')")
parser.add_argument('--image-size', '-imsize', dest='image_size', default=512, type=int, metavar='N',
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                    help="use multiscale vectors for testing, " +
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help="gpu id used for testing (default: '0')")

warnings.filterwarnings("ignore", category=UserWarning)


def tb_setup(save_dir):
    # Setup for tensorboard
    tb_save_dir = os.path.join(
                    save_dir,
                    'summary',
                    )
    if not os.path.exists(tb_save_dir):
        os.makedirs(tb_save_dir)
    
    trash_list = os.listdir(tb_save_dir)
    for entry in trash_list:
        filename = os.path.join(tb_save_dir, entry)
        if fnmatch.fnmatch(entry, '*tfevents*'):
            os.remove(filename)

    summary = SummaryWriter(log_dir=tb_save_dir)

    return summary


def extract_ss(net, _input):
    g_desc, f_desc = net(_input).cpu().data.squeeze()
    return g_desc

def extract_ms(net, _input, ms, msp):

    v = torch.zeros(2048)

    for s in ms:
        if s == 1:
            _input_t = _input.clone()
        else:
            _input_t = torch.nn.functional.interpolate(_input, scale_factor=s, mode='bilinear', align_corners=False)
        
        g_desc, l_desc = net(_input_t)
        g_v = g_desc.pow(msp).cpu().data.squeeze()
        v += g_v

    v /= len(ms)
    v = v.pow(1./msp)
    v /= v.norm()

    return v


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10, mode='test'):
    loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform, mode=mode),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )

    net.cuda()
    net.eval()
    # Extracting Vectors
    with torch.no_grad():
        vecs = torch.zeros(2048, len(images))
        length = len(images)
        with tqdm(total=length) as pbar:
            for i, _input in enumerate(loader):
                _input = _input.cuda()

                if len(ms) == 1 and ms[0] == 1:
                    vecs[:, i] = extract_ss(net, _input)
                else:
                    vecs[:, i] = extract_ms(net, _input, ms, msp)
                
                if (i+1) % print_freq == 0:
                    pbar.update(print_freq)
                elif (i+1) == len(images):
                    pbar.update(len(images) % print_freq)

    return vecs


def main():
    args = parser.parse_args()

    # check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    data_save_path = '/home/user/dataset/'
    download_distractors(data_save_path)

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network
    net = RetrievalNet(backbone='resnet101')
    if args.weights != '':
        state_dicts = torch.load(args.weights, map_location='cpu')
        net.load_state_dict(state_dicts['model'])
        print('>>> Load Weights Done')

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))

    print(">>>> Evaluating scales: {}".format(ms))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        cfg = configdataset(dataset, os.path.join(data_save_path, 'revisitop'))
        images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
        qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None  # for holidaysmanrot and copydays

        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, mode='test')

        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, mode='test')
        qvecs = qvecs.numpy()

        print('>> {}: Evaluating...'.format(dataset))

        vecs_1m = torch.load(args.network + '_vecs_' + 'revisitop1m' + '.pt')
        vecs = torch.cat([vecs, vecs_1m], dim=1)
        vecs = vecs.numpy()

        # search, rank, and print
        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        compute_map_and_print(dataset, ranks, cfg['gnd'])

        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
