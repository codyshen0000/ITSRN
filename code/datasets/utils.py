import os
from PIL import Image
import tqdm
import glob
import numpy as np
import torch

def pngTojpeg(PNG_dir, JPEG_dir, quality=95):
    JPEG_dir = JPEG_dir + f'Q-{quality}'
    os.makedirs(JPEG_dir, exist_ok=True)
    png_names_file = glob.glob(os.path.join(PNG_dir, '*.png'))
    # jpeg_names_file = glob.glob(save_JPEG + '*.jpg')
    for i in tqdm.tqdm(range(len(png_names_file))):
        img = Image.open(png_names_file[i])
        basename = os.path.basename(png_names_file[i])
        img.save(os.path.join(JPEG_dir, basename.replace('.png', '.jpeg')), quality=quality)


def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed    = process_seed - worker_id
    ss  = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))

def numpy_fix_init(worker_id):
    np.random.seed(2<<16 + worker_id)



if __name__ == '__main__':
    png_dir = '/home/lab532/Shen/Dataset/dataset_text/SCTD'
    jpg_dir = '/home/lab532/Shen/Dataset/dataset_text/JPEG/SCTD'

    pngTojpeg(png_dir, jpg_dir, 30)