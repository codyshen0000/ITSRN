import argparse
# import os
from PIL import Image
# import torch
# from torchvision import transforms
# import models





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    # parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    # parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    h, w = list(map(int, args.resolution.split(',')))
    img = Image.open(args.input).convert('RGB')
    
    img.resize((h,w),Image.BILINEAR).save(args.output)