import albumentations as A
import albumentations.pytorch
from core.config import cfg


# train_transform = A.Compose([
#     A.HorizontalFlip(p=0.5), 
#     A.ImageCompression(quality_lower=99, quality_upper=100), 
#     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7), 
#     A.Resize(cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE), 
#     A.Cutout(max_h_size=int(cfg.TRAIN.IM_SIZE * 0.3), 
#              max_w_size=int(cfg.TRAIN.IM_SIZE * 0.3), 
#              num_holes=1, p=0.5),
#     A.Normalize(),
#     A.pytorch.ToTensorV2()
# ])


train_transform = A.Compose([
    A.Resize(cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE), 
    A.HorizontalFlip(),
    A.Normalize(),
    A.pytorch.ToTensorV2()
])


test_transform = A.Compose([
    A.Resize(cfg.TEST.IM_SIZE, cfg.TEST.IM_SIZE), 
    A.Normalize(), 
    A.pytorch.ToTensorV2()
])