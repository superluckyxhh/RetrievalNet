import numpy as np
import math
import cv2

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

_EIG_VALS = np.array([[0.2175, 0.0188, 0.0045]])
_EIG_VECS = np.array(
    [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]
)

def color_norm(im, mean=_MEAN, std=_SD):
    c, h, w = im.shape
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im

def pad(im, size):
    pad_size = ((0, 0), (size, size), (size, size))
    return np.pad(im, pad_size, mode='constant')

def horizon_filp(im, p):
    c, h, w = im.shape
    if np.random.uniform() < p:
        im = im[:, :, ::-1]
    return im

def random_crop(im, size, pad_size=0):
    if pad_size > 0:
        im = pad(im, pad_size)
    h, w = im.shape[1], im.shape[2]
    h_start = np.random.randint(0, h - size)
    w_start = np.random.randint(0, w - size)
    
    im_crop = im[:, h_start:(h_start+size), w_start:(w_start+size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop

def scale(size, im):
    h, w, c = im.shape
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    h_new, w_new = size, size
    if w < h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)

def center_crop(size, im):
    h, w, c = im.shape
    y = int(math.ceil((h - size) / 2))
    x = int(math.ceil((w - size) / 2))
    im_crop = im[y : (y + size), x : (x + size), :]
    assert im_crop.shape[:2] == (size, size)
    return im_crop

def random_size_crop(im, size, area_frac=0.08, max_iter=10):
    h, w, c = im.shape
    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h_crop = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            im_crop = im[y : (y + h_crop), x : (x + w_crop), :]
            assert im_crop.shape[:2] == (h_crop, w_crop)
            im_crop = cv2.resize(im_crop, (size, size), interpolation=cv2.INTER_LINEAR)
            return im_crop.astype(np.float32)
    return center_crop(size, scale(size, im))


def lighting(im, alpha_std=0.1, eig_val=_EIG_VALS, eig_vec=_EIG_VECS):
    """Performs AlexNet-style PCA jitter (CHW format)."""
    if alpha_std == 0:
        return im
    alpha = np.random.normal(0, alpha_std, size=(1, 3))
    alpha = np.repeat(alpha, 3, axis=0)
    eig_val = np.repeat(eig_val, 3, axis=0)
    rgb = np.sum(eig_vec * alpha * eig_val, axis=1)
    for i in range(im.shape[0]):
        im[i] = im[i] + rgb[2 - i]
    return im