import argparse
from core.trainer import train_model, test_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--seed', type=int, default=777)
    args = parser.parse_args()
    
    train_model(args)
    # test_model(args, None)
