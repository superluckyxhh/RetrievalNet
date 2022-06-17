# TODO:

# falcon submit -c 10 -m 10240 -g 4 \
# -p /home/notebook/code/person/ \
# -n Train \
# -r falcon/train.sh

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip setuptools wheel
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple albumentations
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.8.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.2.0.34
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision==0.9.0 --user
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple yacs
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple logzero
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple timm
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX

# TODO:
cd /path/to/model
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py