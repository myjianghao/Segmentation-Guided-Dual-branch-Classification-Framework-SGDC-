from model.Unet import U_Net
from model.Resnet import *
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def dilate(bin_img, ksize=3):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out


class U_Res50(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(U_Res50, self).__init__()
        self.unet = U_Net()
        self.resnet = resnet50(pretrained=pretrained, num_classes=num_classes)
        self.attention_weight = nn.Parameter(torch.ones(640, 896))
        # self.activate = nn.Sigmoid()
        # self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.b = nn.Parameter(torch.tensor(0.0))
        self.activate = nn.Softplus()

    def forward(self, x, is_camloss, s_labels):
        seg_transforms = transforms.Resize((s_labels.shape[-2], s_labels.shape[-1]))
        Sm_outputs = self.unet(seg_transforms(x))
        restore_transforms = transforms.Resize((x.shape[-2], x.shape[-1]))
        # mask_outputs = self.convert_weight.matmul((Sm_outputs > 0.5).float())
        mask_output = self.activate((restore_transforms(Sm_outputs) > 0.5).float())
        mask_output = mask_output*self.attention_weight+self.b
        # mask_output = self.conv1(mask_output) + self.b
        # mask_output =(restore_transforms(Sm_outputs)>0.5).float()
        C_outputs, cam = self.resnet(torch.cat((x, mask_output*x), dim=1), is_camloss)
        return Sm_outputs, C_outputs, cam
