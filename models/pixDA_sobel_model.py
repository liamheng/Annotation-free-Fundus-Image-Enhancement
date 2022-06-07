import torch
import itertools
from .base_model import BaseModel
from . import networks
from models.guided_filter_pytorch.guided_filter import FastGuidedFilter
from models.guided_filter_pytorch.sobel_filter import ThreeSobelFilter, OneSobelFilter
from data.base_dataset import TensorToGrayTensor

class PixDASobelModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_DD', type=float, default=0.2, help='weight for DD')
            parser.add_argument('--lambda_DP', type=float, default=1.0, help='weight for DP')
            parser.add_argument('--lambda_DPG', type=float, default=0.5, help='weight for DPG')
            parser.add_argument('--lambda_G', type=float, default=1.0, help='weight for G loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.input_nc = opt.input_nc

        self.loss_names = ['DP', 'DP_fake', 'DP_real',
                           'DPG', 'DPG_fake', 'DPG_real',
                           'DD', 'DD_fake_SB', 'DD_fake_TB',
                           'G', 'G_DP', 'G_L1', 'G_DPG', 'G_DD']

        self.visual_names = ['real_SA', 'real_SAG', 'fake_SB', 'fake_SBG', 'real_SB', 'real_SBG', 'real_TA', 'real_TAG',
                             'fake_TB', 'fake_TBG']
        # 初始化guide filter和灰度图工具
        if opt.edge_filter == 'guide_filter':
            self.edge_filter = FastGuidedFilter(self.device)
        elif opt.edge_filter == 'one_sobel_filter':
            self.edge_filter = OneSobelFilter(self.device)
        else:
            self.edge_filter = ThreeSobelFilter(self.device)
        self.tensor_to_gray_tensor = TensorToGrayTensor(self.device)

        if self.isTrain:
            self.model_names = ['G', 'DD', 'DP', 'DPG']
        else:  # during test time, only load G
            self.model_names = ['G']

        # 网络的输出是3个channel
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        if self.isTrain:
            self.netDP = networks.define_D(3 * 2, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDPG = networks.define_D((opt.input_nc - 3) * 2, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netDD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                           opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDP.parameters(), self.netDPG.parameters(),
                                                                self.netDD.parameters()), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, isTrain=None):
        """
        处理输入
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_SA = input['SA' if AtoB else 'SB'].to(self.device)
        self.real_SB = input['SB' if AtoB else 'SA'].to(self.device)
        self.real_TA = input['TA' if AtoB else 'TB'].to(self.device)

        self.real_SAG = self.edge_filter(self.tensor_to_gray_tensor(self.real_SA))
        self.real_TAG = self.edge_filter(self.tensor_to_gray_tensor(self.real_TA))
        self.real_SBG = self.edge_filter(self.tensor_to_gray_tensor(self.real_SB))

        self.real_SA6 = torch.cat([self.real_SA, self.real_SAG], dim=1)
        self.real_TA6 = torch.cat([self.real_TA, self.real_TAG], dim=1)
        self.real_SB6 = torch.cat([self.real_SB, self.real_SBG], dim=1)

        self.image_paths = input['TA_path']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_SB = self.netG(self.real_SA6)  # G(SA)
        self.fake_TB = self.netG(self.real_TA6)  # G(TA)

        self.fake_SBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_SB))
        self.fake_TBG = self.edge_filter(self.tensor_to_gray_tensor(self.fake_TB))

        self.fake_SB6 = torch.cat([self.fake_SB, self.fake_SBG], dim=1)
        self.fake_TB6 = torch.cat([self.fake_TB, self.fake_TBG], dim=1)

        self.fake_SAB = torch.cat((self.real_SA, self.fake_SB), dim=1)
        self.real_SAB = torch.cat((self.real_SA, self.real_SB), dim=1)

        self.fake_SABG = torch.cat((self.real_SAG, self.fake_SBG), dim=1)
        self.real_SABG = torch.cat((self.real_SAG, self.real_SBG), dim=1)

    def backward_DD(self):
        """
        Calculate Domain loss for the discriminator, we want to discriminate S and T
        """
        # Fake Target, detach
        pred_fake_SB = self.netDD(self.fake_SB6.detach())
        pred_fake_TB = self.netDD(self.fake_TB6.detach())

        self.loss_DD_fake_SB = self.criterionGAN(pred_fake_SB, True)
        self.loss_DD_fake_TB = self.criterionGAN(pred_fake_TB, False)

        # combine loss and calculate gradients
        self.loss_DD = (self.loss_DD_fake_SB + self.loss_DD_fake_TB) * 0.5
        self.loss_DD.backward()

    def backward_DP(self):
        """
        Calculate GAN loss for the discriminator
        """
        pred_fake_SAB = self.netDP(self.fake_SAB.detach())
        pred_real_SAB = self.netDP(self.real_SAB.detach())

        self.loss_DP_fake = self.criterionGAN(pred_fake_SAB, False)
        self.loss_DP_real = self.criterionGAN(pred_real_SAB, True)

        # combine loss and calculate gradients
        self.loss_DP = (self.loss_DP_fake + self.loss_DP_real) * 0.5
        self.loss_DP.backward()

    def backward_DPG(self):
        """
        Calculate GAN loss for the discriminator
        """
        pred_fake_SABG = self.netDPG(self.fake_SABG.detach())
        pred_real_SABG = self.netDPG(self.real_SABG.detach())

        self.loss_DPG_real = self.criterionGAN(pred_real_SABG, True)
        self.loss_DPG_fake = self.criterionGAN(pred_fake_SABG, False)

        self.loss_DPG = (self.loss_DPG_fake + self.loss_DPG_real) * 0.5
        self.loss_DPG.backward()


    def backward_G(self):
        """
        Calculate GAN and L1 loss for the generator
        Generator should fool the DD and DP
        """
        # First, G(A) should fake the discriminator
        pred_fake_SAB = self.netDP(self.fake_SAB)
        pred_fake_SABG = self.netDPG(self.fake_SABG)

        self.loss_G_DP = self.criterionGAN(pred_fake_SAB, True) * self.opt.lambda_DP
        self.loss_G_DPG = self.criterionGAN(pred_fake_SABG, True) * self.opt.lambda_DPG

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_SB, self.real_SB) * self.opt.lambda_L1

        # Third,  G(SA) and G(TA) should fool the domain discriminator
        pred_fake_TB = self.netDD(self.fake_TB6)
        pred_fake_SB = self.netDD(self.fake_SB6)
        self.loss_G_DD = (self.criterionGAN(pred_fake_TB, True) + self.criterionGAN(pred_fake_SB, False)) * \
                         0.5 * self.opt.lambda_DD

        self.loss_G = self.loss_G_DP + self.loss_G_DPG + self.loss_G_L1 + self.loss_G_DD
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update DD (domain discriminator)
        self.set_requires_grad([self.netDP, self.netDPG, self.netDD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_DP()      # calculate gradients for D_A
        self.backward_DPG()      # calculate graidents for D_B
        self.backward_DD()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

        # update G
        self.set_requires_grad([self.netDP, self.netDPG, self.netDD],
                               False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

