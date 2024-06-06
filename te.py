import cv2 as cv
from PIL.Image import Image
from einops import rearrange
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn import Conv2d, Sequential, InstanceNorm2d, ReLU, Linear

import cv2
# coding:utf8
import cv2
import numpy as np
import matplotlib.pyplot as plt


#########Rio
# img = cv2.imread('D:\\CEDp\\datasets\Apple\\63_100.jpg')
# #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
# img2_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# ret,ma1=cv2.threshold(img2_gray,170,255,cv2.THRESH_BINARY)
# cv2.imshow('1',ma1)
#
#
#
# plt.subplot(131), plt.imshow(ma1)
# plt.title('CV2 Image1'), plt.xticks([]), plt.yticks([])
# plt.show()

# img = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
# img1 = cv.imread("D:\\CEDp\\datasets\\0184")
# img2not = cv.bitwise_not(img1)
#########

######################## Sobel #############################
# def sob(feat):
#
#     kernel_dx = [[1, 0, 1],
#                  [-2, 0, 2],
#                  [-1, 0, 1]]
#
#     kernel_dy = [[1, -2, -1],
#                  [0, 0, 0],
#                  [1, 2, 1]]
#
#     dim = feat.size()[1]
#
#     dx_window = torch.FloatTensor(kernel_dx).expand(dim,dim, 3, 3)
#     dx_weight = nn.Parameter(data=dx_window, requires_grad=False).cuda()
#
#     dy_window = torch.FloatTensor(kernel_dy).expand(dim,dim, 3, 3)
#     dy_weight = nn.Parameter(data=dy_window, requires_grad=False).cuda()
#
#     return F.conv2d(feat, dx_weight, padding=1), F.conv2d(feat, dy_weight, padding=1)
#
#
# if __name__ == "__main__":
#     x =torch.randn(3,8,8,8).cuda()
#
#     x1, y1 = sob(x)
#
#     print(x1)
#     print(y1)
 ############################################################################
# temperature = nn.Parameter(torch.ones(2, 1, 1))
# k = torch.randn(1,4,4,4)
# v = torch.randn(1,4,4,4)
# b, d, h, w = k.size()
# a = torch.tensor([0.2])
# attn_a = torch.nn.Parameter(a, requires_grad=True)
# ki = rearrange(k, 'b (head d) h w -> b head d (h w)', head=2)
# vi = rearrange(v, 'b (head d) h w -> b head d (h w)', head=2)
# _, _, D, _ = ki.size()
# mask1 = torch.zeros(b, 2, D, D)
# r_attn = (ki @ vi.transpose(-2, -1)) * temperature
# index = torch.topk(r_attn, k=int(D / 2 ), dim=-1, largest=True)[1]
# mask1.scatter_(-1, index, 1.)
# print('mask1', mask1)
# attn1 = torch.where(mask1 > 0, r_attn, torch.full_like(r_attn, float('-inf')))
# print('attn1', attn1.shape)
# attn1 = attn1.softmax(dim=-1)
# out = attn1 @ vi
# print('out', out.shape)
# ut = out * attn_a
# print('ut', ut.shape)


# b, c, h, w = k.size()
#
# v = torch.randn(1,4,4,5)
# k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=2)
# v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=2)
#
# _, _, C, _ = k.shape
#
# print(k.shape)
#
# attn = k @ v.transpose(-2,-1)
# print('attn', attn.shape)  # s head 2 g
# top_k = torch.topk(attn,k=int(C/2),dim=-1)[1]
# print('top_k',top_k.shape)
#
# n, head, c_kv, hw,  = v.size()
# 'b head c (h w)'
#
# topk = top_k.size(-1)
# top_v = torch.gather(v.view(n, 1, head, c_kv, hw).expand(-1, head, -1, -1, -1),
#                      dim=-1,
#                      index=topk.view(n, head, 1, c_kv, topk).expand(-1, -1, head, -1, -1))
#
# print('top_v', top_v.shape)



# from torch.nn import MaxPool2d,MaxUnpool2d
# x = torch.randn(1,1,4,4) #HF
#
# qx = rearrange(x, 'b c (h head) w -> b c head (h w)', head=2)
#
# print("qx",qx)
#
# s = torch.randn(1,1,4,4) # LF
#
# qs = rearrange(s, 'b c (h head) w -> b c head (h w)', head=2)
# print("qs",qs)
# # 1,1,2,8
# # ####取高频的最大关系#######
# pool = MaxPool2d(2, return_indices=True)
# unPool = MaxUnpool2d(2)
# #
# y, ind = pool(qx)
# #
# high_x = unPool(y, ind)
#
# print("恢复值high_x：",high_x)
#
# print(ind)
#
# xs =qs * high_x + qs
#
# print("选取的最大值xs",xs)

# To be specific, we introduce a novel Luminance-Structure Guidance Head (LS-GH) which not only compute the grey-weighted difference between the feature vectors and their neighbourhoods to obtain structural map, but also employ multi-channel curve fine-tuning to correct distribution of the gray-level. The output of LS-GH as a condition to guide diffusion-generation process leading sharp and realistic results ; Yet, the structural information in our model is implemented as the edge detection in image space. Furthermore, to avoid amplifying noise or over-smoothing the output during feature enhancement, we propose Sparse Attention Enhancement Module (SAEM) which can be easily plugged into conventional U-Net to highlight the local neighborhood that contains the most useful fine-grained representations.
import torch
from PIL import Image

# 读取四通道图像
img = Image.open('D:\\CEDp\\demo-frames\\SDSD\\0175.png')
print(img.shape)
# # 将四通道图像转换为三通道
# img = img.convert('RGB')
#
# # 将PIL Image转换为torch.Tensor
# img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float().div(255)