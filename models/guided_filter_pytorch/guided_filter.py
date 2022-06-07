from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# import kornia
import torch
import torchvision.transforms as transforms
from .box_filter import BoxFilter

class FastGuidedFilter(nn.Module):
    def __init__(self, device, r=3, eps=0.003):
        super(FastGuidedFilter, self).__init__()
        self.device =device
        self.r = r
        self.eps = eps
        self.eps_list = [0.005, 0.003, 0.001]
        # self.w_h_size = w_h_size
        # self.boxfilter = BoxFilter(r)
        self.mean_kernel = torch.ones(1, 1, 3, 3).to(self.device) / 9
        self.mean_kernel.requires_grad = False
        self.padding = nn.ReplicationPad2d(1)
        # self.one = torch.ones([1, 1, w_h_size, w_h_size]).to(self.device)
        # self.N = self.boxfilter(self.one)
        # self.epss = 1e-12
        self.transforms = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def forward(self, I):
        # TODO:不知道是不是正确的mean filter
        I = self.padding(I)
        mean_I = F.conv2d(I, self.mean_kernel)
        mean_II = F.conv2d(I * I, self.mean_kernel)
        var_I = mean_II - mean_I * mean_I
        a1 = var_I / (var_I + self.eps_list[0])
        mean_a1 = F.conv2d(self.padding(a1), self.mean_kernel)
        a2 = var_I / (var_I + self.eps_list[1])
        mean_a2 = F.conv2d(self.padding(a2), self.mean_kernel)
        a3 = var_I / (var_I + self.eps_list[2])
        mean_a3 = F.conv2d(self.padding(a3), self.mean_kernel)

        mean_a = torch.cat([mean_a1, mean_a2, mean_a3], dim=1)
        for i in range(len(mean_a)):
            mean_a[i] = self.transforms(mean_a[i])
        # mean_a = torch.cat([I, I, I], dim=1)
        return mean_a

