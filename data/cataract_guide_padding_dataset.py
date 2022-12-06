import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_params, get_transform_six_channel
from data.image_folder import make_dataset
from PIL import Image


class CataractGuidePaddingDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_source = os.path.join(opt.dataroot, 'source')  # get the image directory
        self.dir_target = os.path.join(opt.dataroot, 'target')  # get the image directory
        self.dir_source_mask = os.path.join(opt.dataroot, 'source_mask')  # get the image directory
        self.dir_target_mask = os.path.join(opt.dataroot, 'target_mask')  #

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))  # get image paths
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))  # get image paths
        self.source_mask_paths = sorted(make_dataset(self.dir_source_mask, opt.max_dataset_size))  # get image paths
        self.target_mask_paths = sorted(make_dataset(self.dir_target_mask, opt.max_dataset_size))  # get image paths

        self.target_size = len(self.target_paths)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        source_path = self.source_paths[index]
        source_path_mask_path = os.path.join(self.dir_source_mask, os.path.split(source_path)[-1].replace('jpg', 'png'))
            # self.source_mask_paths[index]
        target_index = random.randint(0, self.target_size - 1) if self.isTrain else index % self.target_size
        target_path = self.target_paths[target_index]
        target_mask_path = self.target_mask_paths[target_index]

        SAB = Image.open(source_path).convert('RGB')
        TA = Image.open(target_path).convert('RGB')
        SA_mask = Image.open(source_path_mask_path).convert('L')
        SB_mask = SA_mask
        TA_mask = Image.open(target_mask_path).convert('L')
        w, h = SAB.size
        w2 = int(w / 2)
        SA = SAB.crop((0, 0, w2, h))
        SB = SAB.crop((w2, 0, w, h))


        # 对输入和输出进行同样的transform（裁剪也继续采用）
        source_transform_params = get_params(self.opt, SA.size)
        source_A_transform, source_A_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        source_B_transform, source_B_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.output_nc == 1))
        #  = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.input_nc == 1))
        # source_B_mask_transform = get_transform_six_channel(self.opt, source_transform_params, grayscale=(self.output_nc == 1))

        target_transform_params = get_params(self.opt, TA.size)
        target_A_transform, target_A_mask_transform = get_transform_six_channel(self.opt, target_transform_params, grayscale=(self.input_nc == 1))

        SA = source_A_transform(SA)
        S_mask = source_A_mask_transform(SA_mask)
        # 使用同一个transform
        SB = source_B_transform(SB)
        # SB_mask = source_B_mask_transform(SB_mask)

        TA = target_A_transform(TA)
        T_mask = target_A_mask_transform(TA_mask)

        return {'SA': SA, 'SB': SB, 'S_mask': S_mask, 'SA_path': source_path,
                'SB_path': source_path, 'TA': TA, 'T_mask': T_mask, 'TA_path': target_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.source_paths)
