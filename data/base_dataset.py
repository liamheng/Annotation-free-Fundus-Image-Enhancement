"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F
# from torch.autograd import Variable
# # import kornia
import torch
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size, is_source=True):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        # TODO:添加数组的尺寸来随机挑选，target只随机裁286
        if opt.source_size_count == 1:
            new_h = new_w = opt.load_size
        else:
            if not opt.isTrain:
                new_h = new_w = opt.load_size
            else:
                new_h = new_w = random.choice([286, 306, 326, 346])
            # new_h = new_w = random.choice([opt.load_source_size, opt.load_target_size])
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    flip_vertical = random.random() > 0.5

    return {'load_size': new_h, 'crop_pos': (x, y), 'flip': flip, 'flip_vertical': flip_vertical}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        # TODO:resize需要优化，此处只考虑params没有时直接取target的
        if params is None:
            osize = [opt.load_size, opt.load_size]
        else:
            load_size = params['load_size']
            osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 加入上下翻转
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomVerticalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform_six_channel(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    mask_transform_list = []
    if 'resize' in opt.preprocess:
        if params is None:
            osize = [opt.load_size, opt.load_size]
        else:
            load_size = params['load_size']
            osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
        mask_transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
        mask_transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
            mask_transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
            mask_transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
        mask_transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
            mask_transform_list.append(transforms.RandomHorizontalFlip())

        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
            mask_transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 加入上下翻转
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomVerticalFlip())
            mask_transform_list.append(transforms.RandomVerticalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
            mask_transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
    if convert:
        transform_list += [transforms.ToTensor()]
        mask_transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list), transforms.Compose(mask_transform_list)


def get_gray_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    gray_transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if 'resize' in opt.preprocess:
        if params is None:
            osize = [opt.load_size, opt.load_size]
        else:
            load_size = params['load_size']
            osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
        gray_transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 加入上下翻转
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomVerticalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
    gray_transform_list.append(transforms.Grayscale(1))
    gray_transform_list += transform_list
    if convert:
        transform_list += [transforms.ToTensor()]
        gray_transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
            # gray_transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            # gray_transform_list += [transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(transform_list), transforms.Compose(gray_transform_list)


class TensorToGrayTensor(nn.Module):
    def __init__(self, device, R_rate=0.299, G_rate=0.587, B_rate=0.114):
        super(TensorToGrayTensor, self).__init__()
        self.kernel = torch.tensor([])
        self.kernel = torch.empty(size=(1, 3, 1, 1), dtype=torch.float32, device=device)
        self.kernel.requires_grad = False
        # TODO:确定输入是RGB
        self.kernel[0, 0, 0, 0] = R_rate
        self.kernel[0, 1, 0, 0] = G_rate
        self.kernel[0, 2, 0, 0] = B_rate

    def forward(self, x):
        output = F.conv2d(x, self.kernel)
        return output


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __flip_vertical(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
