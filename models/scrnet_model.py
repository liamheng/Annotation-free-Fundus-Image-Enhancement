# -*- coding: UTF-8 -*-
"""
@Function:from two-stage to one-stage
@File: DG_one_model.py
@Date: 2021/9/14 20:45 
@Author: Hever
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.HFC_filter import HFCFilter


def hfc_mul_mask(hfc_filter, image, mask):
    hfc = hfc_filter(image, mask)
    # return hfc
    return (hfc + 1) * mask - 1
    # return image


class SCRNetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='unet_combine_2layer', dataset_mode='aligned', no_dropout=True, lr=0.0002)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0)
            parser.add_argument('--lambda_L1H', type=float, default=1.0)
            parser.add_argument('--lambda_L1_HFC', type=float, default=1.0)
        parser.add_argument('--filter_width', type=int, default=27, help='weight for G loss')
        parser.add_argument('--nsig', type=int, default=9, help='weight for G loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1H', 'G_L1', 'G_L1_HFC', 'G']

        self.visual_names_train = ['real_SA', 'real_SAH', 'fake_SBH', 'fake_SB', 'fake_SB_HFC',
                                   'real_SB', 'real_SBH', ]
        self.visual_names_test = ['real_TA', 'real_TAH', 'fake_TBH', 'fake_TB', 'fake_TB_HFC']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            self.visual_names = self.visual_names_train
        else:
            self.model_names = ['G']
            self.visual_names = self.visual_names_test

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.hfc_filter = HFCFilter(opt.filter_width, opt.nsig, sub_low_ratio=1, sub_mask=True, is_clamp=True).to(self.device)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input, isTrain=None):
        """
        set the input
        """
        AtoB = self.opt.direction == 'AtoB'
        if not self.isTrain or isTrain is not None:
            self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)
            self.T_mask = input['T_mask'].to(self.device)
            self.real_TAH = hfc_mul_mask(self.hfc_filter, self.real_TA, self.T_mask)
            self.image_paths = input['TA_path']
        else:
            self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
            self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
            self.S_mask = input['S_mask'].to(self.device)
            self.image_paths = input['SA_path']
            self.real_SAH = hfc_mul_mask(self.hfc_filter, self.real_SA, self.S_mask)
            self.real_SBH = hfc_mul_mask(self.hfc_filter, self.real_SB, self.S_mask)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_SBH, self.fake_SB = self.netG(self.real_SAH)
        self.fake_SBH = (self.fake_SBH + 1) * self.S_mask - 1
        self.fake_SB = (self.fake_SB + 1) * self.S_mask - 1
        self.fake_SB_HFC = hfc_mul_mask(self.hfc_filter, self.fake_SB, self.S_mask)


    def test(self):
        self.visual_names = self.visual_names_test
        with torch.no_grad():
            # For visualisation
            self.fake_TBH, self.fake_TB = self.netG(self.real_TAH)
            self.fake_TBH = (self.fake_TBH + 1) * self.T_mask - 1
            self.fake_TB = (self.fake_TB + 1) * self.T_mask - 1
            self.fake_TB_HFC = hfc_mul_mask(self.hfc_filter, self.fake_TB, self.T_mask)
            # self.fake_TBH = self.hfc_filter(self.fake_TB_HFC, self.T_mask)

            self.compute_visuals()

    def train(self):
        """Make models eval mode during test time"""
        self.visual_names = self.visual_names_train
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()


    def backward_G(self):
        # LR
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1
        # LH
        self.loss_G_L1H = self.criterionL1(self.fake_SBH, self.real_SBH) * self.opt.lambda_L1H
        # ！！！Lcyc
        self.loss_G_L1_HFC = self.criterionL1(self.fake_SB_HFC, self.fake_SBH.detach()) * self.opt.lambda_L1_HFC

        self.loss_G = self.loss_G_L1 + self.loss_G_L1H + self.loss_G_L1_HFC
        self.loss_G.backward()


    def optimize_parameters(self):
        # self.set_requires_grad([self.netG], True)  # D requires no gradients when optimizing G
        self.forward()                   # compute fake images: G(A)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

