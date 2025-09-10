# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import math
import warnings
from PIL import Image
import glob
from tool.utils import load_network
from torchvision import  transforms

warnings.filterwarnings("ignore")

# 定义所有参数（替代opts.yaml）
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids', default='2', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--drone_dir', default='/media/zeh/38421e79-4336-4036-a3ab-8b8d0916308d/sc/DenseUAV-main-v1/checkpoints/test/query_drone/000000', type=str, help='无人机图像目录')
parser.add_argument('--satellite_dir', default='/media/zeh/38421e79-4336-4036-a3ab-8b8d0916308d/sc/DenseUAV-main-v1/checkpoints/test/gallery_satellite/000000', type=str, help='卫星图像目录')
parser.add_argument('--output_file', default='similarities.txt', type=str, help='相似度输出文件')
parser.add_argument('--checkpoint', default='net_119.pth', type=str, help='模型权重路径')
parser.add_argument('--batchsize', default=32, type=int, help='批处理大小')
parser.add_argument('--h', default=224, type=int, help='图像高度')
parser.add_argument('--w', default=224, type=int, help='图像宽度')
parser.add_argument('--mode', default=1, type=int, help='1:drone->satellite  2:satellite->drone')
parser.add_argument('--backbone', default='ViTB-224', type=str, help='骨干网络类型')
parser.add_argument('--head', default='FSRA', type=str, help='头部网络类型')
parser.add_argument('--block', default=1, type=int, help='块数量')
parser.add_argument('--nclasses', default=2256, type=int, help='类别数量')
parser.add_argument('--in_planes', default=768, type=int, help='输入维度')
parser.add_argument('--num_bottleneck', default=224, type=int, help='瓶颈层维度')
parser.add_argument('--droprate', default=0.5, type=int, help='0.5')
parser.add_argument('--load_from', default='no', type=str, help='no')

opt = parser.parse_args()

# GPU设置
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
    device = torch.device(f'cuda:{gpu_ids[0]}')
else:
    device = torch.device('cpu')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()
model = load_network(opt)
print("这是%s的结果" % opt.checkpoint)
# model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()


def extract_features(images):
    """提取图像特征"""
    features = []
    num_images = len(images)

    # 分批处理
    for i in range(0, num_images, opt.batchsize):
        batch_images = images[i:i + opt.batchsize]
        batch_tensors = torch.stack([transform(img) for img in batch_images]).to(device)

        with torch.no_grad():
            if opt.mode == 1:  # drone->satellite
                _, outputs = model(None, batch_tensors)
            else:  # satellite->drone
                outputs, _ = model(batch_tensors, None)

            feat = outputs[1]  # 获取特征向量

        # 特征归一化
        fnorm = torch.norm(feat, p=2, dim=1, keepdim=True)
        feat = feat.div(fnorm.expand_as(feat))
        features.append(feat.cpu())

    return torch.cat(features, dim=0)


# 加载无人机图像
drone_files = glob.glob(os.path.join(opt.drone_dir, '*.JPG')) + glob.glob(os.path.join(opt.drone_dir, '*.png'))
if not drone_files:
    raise FileNotFoundError(f"No images found in {opt.drone_dir}")

drone_img = Image.open(drone_files[0]).convert('RGB')
drone_feat = extract_features([drone_img])[0].unsqueeze(0)  # 保持二维形状 [1, feat_dim]

# 加载卫星图像
satellite_files = glob.glob(os.path.join(opt.satellite_dir, '*.jpg')) + glob.glob(
    os.path.join(opt.satellite_dir, '*.png')) + glob.glob(
    os.path.join(opt.satellite_dir, '*.tif'))
if not satellite_files:
    raise FileNotFoundError(f"No images found in {opt.satellite_dir}")

satellite_imgs = [Image.open(f).convert('RGB') for f in satellite_files]
satellite_feats = extract_features(satellite_imgs)

# 计算相似度
similarities = torch.mm(satellite_feats, drone_feat.t()).squeeze(1).numpy()

# 保存结果
with open(opt.output_file, 'w') as f:
    f.write("satellite_path,similarity\n")
    for path, sim in zip(satellite_files, similarities):
        f.write(f"{path},{sim:.6f}\n")

print(f"相似度计算完成! 结果保存至 {opt.output_file}")
print(f"无人机图像: {drone_files[0]}")
print(f"处理卫星图像: {len(satellite_files)}张")
print(f"最大相似度: {np.max(similarities):.4f}, 最小相似度: {np.min(similarities):.4f}")