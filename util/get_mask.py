# -*- coding: UTF-8 -*-
"""
@Function: 给source和target的做一个mask
@File: get_mask.py
@Date: 2021/6/10 20:50
@Author: Hever
"""
import os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from optparse import OptionParser


def get_mask(img):
    gray = np.array(img.convert('L'))
    return ndimage.binary_opening(gray > 10, structure=np.ones((8, 8)))


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--image_dir', default='./images/drive_cataract/source',
                      help="input directory to the source image.")
    parser.add_option('--output_dir', default='./images/drive_cataract/source_mask',
                      help="output directory to the source mask.")
    parser.add_option('--mode', default='pair',
                      help="pair option is for the source image, single option is for the target image")
    (opt, args) = parser.parse_args()
    image_dir = opt.image_dir
    output_dir = opt.output_dir

    mkdir(output_dir)
    image_list = os.listdir(image_dir)
    for image_name in image_list:
        image_path = os.path.join(image_dir, image_name)
        output_path_A = os.path.join(output_dir, image_name.split('.')[0] + '.png')

        if opt.mode == 'pair':
            SAB = Image.open(image_path).convert('RGB')
            w, h = SAB.size
            w2 = int(w / 2)
            SA = SAB.crop((0, 0, w2, h))
            image = SAB.crop((w2, 0, w, h))
        else:
            image = Image.open(image_path).convert('RGB')

        mask = get_mask(image)
        cv2.imwrite(output_path_A, mask * 255)