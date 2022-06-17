import argparse
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import json
from PIL import Image as pImage
from torchvision import models
from time import time
from tqdm import tqdm
from torchvision import transforms

from revisitop.genericdataset import ImagesFromList



def imagenet_prediction(model, in_tensor):
    model.eval()
    in_tensor = in_tensor.cuda()
    with torch.no_grad():
        out_tensor = model(in_tensor)
    # pred = torch.max(out_tensor, 1)[1].cpu().numpy()[0]
    feature = out_tensor.cpu().numpy()
    return feature


def image_retrieval_by_classification(model, in_tensor,
                                      label_image_dict, topk=5):
    # get image classification prediction
    prediction = imagenet_prediction(model, in_tensor)
    topk_preds = np.argsort(prediction[0])[-topk:][::-1]
    topk_img_index_list = []
    for idx in topk_preds:
        topk_img_index_list += label_image_dict[str(idx)]
    return np.array(topk_img_index_list)


def prepare_image_input(image_path, transform, transform1):
    img_pil = pImage.open(image_path).convert('RGB')
    cls_input = transform(img_pil)
    cls_input = torch.unsqueeze(cls_input, 0)
    img_pil.thumbnail((1024, 1024), pImage.ANTIALIAS)
    solar_input = transform1(img_pil)
    solar_input = torch.unsqueeze(solar_input, 0)
    return cls_input, solar_input


def load_image_click(img_path, transform, transform1):
    global cls_input, solar_input
    cls_input, solar_input = prepare_image_input(img_path, transform, transform1)
    print('loading image from {}'.format(img_path))
    img_data = open(img_path, 'rb').read()
    
    return img_data, cls_input, solar_input


def get_img_feature(net, _input, ms):
 #   t_vecs = extract_vectors(net, img_path, 1024, transform, ms=ms, mode='test', show_progress=False)
    t_vecs = extract_vector_from_one_image(net, _input, ms)
    t_vecs = t_vecs.numpy()
    return t_vecs


def score_and_rank_from_dot_product(vecs, t_vecs, k=None):
    scores = np.dot(vecs.T, t_vecs)
    ranks = np.argsort(-scores, axis=0)
    if k is None:
        return ranks, scores
    return ranks, scores, ranks[:k, :]


def topk_retrieval(ranks, scores, objs, k):
    topk_objs = []
    topk_scores = []
    for rank in ranks:
        obj = objs[rank]
        if obj not in topk_objs:
            topk_objs.append(obj)
            topk_scores.append(scores[rank])
        if len(topk_objs) >= k:
            return topk_objs, topk_scores
        
        
def get_retrieval_res(enet, net, label_image_dict, 
                      features, obj_list, images, ms
    ):
    model_dir = '/apdcephfs/share_782420/jiayudong/datasets/sketchfab_gltf_unzip/'
    print('Extracting image feature...')
    s = time()
    image_indices = image_retrieval_by_classification(enet, cls_input, label_image_dict, topk=1)
    s1 = time()
    print('get features using imagenet label time: {}'.format(s1-s))
    print('Retrieving top5 models from gallery...')
    # get topk obj retrieval features
    t_features = features[:, image_indices]

    t_obj_list = [obj_list[i] for i in image_indices]
    t_img_list = [images[i] for i in image_indices]
    
    t_vec = get_img_feature(net, solar_input, ms)
    s2 = time()
    print('retrieval feature extraction time: {}'.format(s2-s1))
    
    ranks, scores = score_and_rank_from_dot_product(t_features, t_vec)
    ranks, scores = ranks[:, 0], scores[:, 0]
    topk_objs, topk_scores = topk_retrieval(ranks, scores, t_obj_list, 5)
    s3 = time()
    print('model retrieval time:{}'.format(s3-s2))
    topk_paths = [os.path.join(model_dir, obj, 'scene_2.gltf') for obj in topk_objs]
    
    # Test model 1
    
    # model0_path = topk_paths[0]
    # gltf = GLTF.load(model0_path)
    # print(gltf.model)

    # print(gltf.model.buffers[0].uri)
    # print(gltf.resources)
    # resource = gltf.resources[0]
    # print(resource)
    
    return topk_paths

# def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, mode='test'):
#     loader = torch.utils.data.DataLoader(
#             ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs,
#             transform=transform, mode=mode),
#             batch_size=1, shuffle=False, num_workers=8, pin_memory=True
#         )

#     # Extract Vectors
#     with torch.no_grad():
#         vecs = torch.zeros(2048, len(images))
#         length = len(images)
#         with tqdm(total=length) as pbar:
#             for i, _input in enumerate(loader):
#                 _input = _input.cuda()

#                 if len(ms) == 1 and ms[0] == 1:
#                     vecs = net(_input).cpu().data.squeeze()
#                 else:
#                     v = torch.zeros(2048)
#                     for s in ms:
#                         if s == 1:
#                             _input_t = _input.clone()
#                         else:
#                             _input_t = torch.nn.functional.interpolate(_input, scale_factor=s, mode='bilinear', align_corners=False)
#                         v += net(_input_t).pow(1).cpu().data.squeeze()

#                     v /= len(ms)
#                     v = v.pow(1./1)
#                     vecs /= v.norm()
#     return vecs


def extract_retrieval_run(q_ims_path):
    parser = argparse.ArgumentParser(description='Example Script for extracting and saving descriptors for R-1M disctractors')
    parser.add_argument('--network', '-n', metavar='NETWORK', default='resnet101-solar-best.pth', 
                        help="network to be evaluated. ")
    parser.add_argument('--image-size', '-imsize', dest='image_size', 
                        default=256, type=int, metavar='N',
                        help="maximum size of longer image side used for testing (default: 1024)")
    parser.add_argument('--multiscale', '-ms', metavar='MULTISCALE', default='[1, 2**(1/2), 1/2**(1/2)]',
                        help="use multiscale vectors for testing, " +
                        " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1]')")
    parser.add_argument('--soa', action='store_true',
                        help='use soa blocks')
    parser.add_argument('--soa-layers', type=str, default='45',
                        help='config soa blocks for second-order attention')

    # GPU ID
    parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                        help="gpu id used for testing (default: '0')")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    net = load_network(network_name=args.network)
    net.mode = 'test'

    print(">>>> loaded network: ")
    print(net.meta_repr())

    ms = list(eval(args.multiscale))
    print(">>>> Evaluating scales: {}".format(ms))

    net.cuda()
    net.eval()
    

    enet = models.efficientnet_b7(pretrained = True)
    enet = enet.cuda()
    # For cls input
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
    # For sloar input
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

          
    # 3D Database
    print('Load image list and features...')
    json_path = '/apdcephfs/share_782420/jiayudong/datasets/sketchfab_info/sketchfab_pyvista_render_list.json'
    
    with open(json_path, 'r') as file:
        meta_data = json.load(file)
    image_list = meta_data['img_list']
    obj_list = meta_data['obj_list']
    with open('/apdcephfs/share_782420/jiayudong/datasets/sketchfab_info/enet_label_image_dict.json', 'r') as file:
        label_image_dict = json.load(file)
    image_dir = '/apdcephfs/share_782420/jiayudong/datasets/'
    images = [os.path.join(image_dir, img) for img in image_list]
    feature_path = '/apdcephfs/share_782420/jiayudong/datasets/sketchfab_info/sketchfab_pyvista_render_vecs.npy'
    features = np.load(feature_path)
    
    
    querys = [os.path.join(q_ims_path, n) for n in os.listdir(q_ims_path)]
    querys = querys[0]
    
    im, cls_in, sloar_in = load_image_click(querys, transform, transform1)
    topk_retrieval_model_path = get_retrieval_res(enet=enet, net=net, label_image_dict=label_image_dict,
                      features=features, obj_list=obj_list, images=images,
                      ms=ms)
   
    return topk_retrieval_model_path