import cv2
import time
import torch
import torchvision
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import rotate
import scipy.ndimage as ndimage
from PIL import Image
import PFtool


def rrp(img, angle, resize):
    height, width = img.shape[:2]  # 地图宽高
    if height > width:
        cut = width//(1.414*2)
    else:
        cut = height // (1.414 * 2)
    img = img + 1
    rotated_matrix = torchvision.transforms.functional.rotate(img.unsqueeze(0), float(angle))
    rotated_matrix = rotated_matrix[:, int(height//2-cut):int(height//2+cut), int(width//2-cut):int(width//2+cut)]
    resized_matrix = transforms.Resize((resize, resize), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(rotated_matrix.unsqueeze(0)).squeeze(0)
    return resized_matrix

def rrp_DL(img, angle, resize):
    height, width = img.shape[1], img.shape[2]  # 地图宽高
    if height > width:
        cut = width//(1.414*2)
    else:
        cut = height // (1.414 * 2)
    rotated_matrix = torchvision.transforms.functional.rotate(img, float(angle))
    rotated_matrix = rotated_matrix[:, int(height//2-cut):int(height//2+cut), int(width//2-cut):int(width//2+cut)]
    resized_matrix = transforms.Resize((resize, resize), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(rotated_matrix.unsqueeze(0)).squeeze(0)
    return resized_matrix


def rrp_numpy(img, angle, resize):
    x, y = img.shape[0:2]
    img = cv2.resize(img, (int(y / 4), int(x / 4)))
    height, width = img.shape[:2]  # 地图宽高
    if height > width:
        cut = width//(1.414*2)
    else:
        cut = height // (1.414 * 2)
    rotated_img = ndimage.rotate(img, angle, reshape=False)
    rotated_matrix = rotated_img[int(height // 2 - cut):int(height // 2 + cut), int(width // 2 - cut):int(width // 2 + cut)]
    resized_matrix = cv2.resize(rotated_matrix, (resize, resize))
    return resized_matrix


def get_uav_rotate_DL(uav_img, device, resize):
    uav_g = torch.from_numpy(uav_img).to(device).permute(2, 0, 1)    # torch BGR (3,w,h)
    uav_rotate = []
    for i in range(25):
        uav_view_rotated = rrp_DL(uav_g, 3.6 * i, resize).squeeze(0)
        uav_rotate.append(uav_view_rotated)
    for k in range(1, 4):
        for i in range(25):
            uav_view_rotated = torch.rot90(uav_rotate[i], k=k, dims=(1, 2))
            uav_rotate.append(uav_view_rotated)
    show = 0
    if show:
        for i in range(100):
            rotated_numpy = np.transpose(uav_rotate[i].cpu().numpy().astype(np.uint8), (1, 2, 0))  # numpy BGR (W,H,3)
            cv2.imshow('1111', rotated_numpy)  # 转换回 BGR 格式显示
            cv2.waitKey(0)

            # show_np_pic(uav_rotate[i].permute(1, 2, 0).numpy().astype(np.uint8), "show_uav_and_goal_view", 1)  # 显示矩阵照片
            # cv2.waitKey(0)
    return uav_rotate  # torch BGR (3,w,h)


def get_uav_rotate_g(uav_img, device, resize):
    uav_g = torch.from_numpy(uav_img).to(device)
    part_time_start = time.time()
    uav_rotate = []
    for i in range(25):
        uav_view_rotated = rrp(uav_g, 3.6 * i, resize).squeeze(0)  # 无人机旋转矩阵 rrp自带加1
        uav0 = torch.where(uav_view_rotated != 0 + 1, 0, uav_view_rotated)  # 小路
        uav1 = torch.where(uav_view_rotated != 1 + 1, 0, uav_view_rotated) // 2 + torch.where(uav_view_rotated != 5 + 1, 0, uav_view_rotated) // 6  # 建筑
        uav2 = torch.where(uav_view_rotated != 2 + 1, 0, uav_view_rotated) // 3  # 马路
        uav3 = torch.where(uav_view_rotated != 3 + 1, 0, uav_view_rotated) // 4  # 植被
        uav_view_rotated = torch.cat((uav0.unsqueeze(0), uav1.unsqueeze(0), uav2.unsqueeze(0), uav3.unsqueeze(0)), dim=0)
        # uav_view_rotated = torch.cat((uav0.unsqueeze(0), uav1.unsqueeze(0), uav2.unsqueeze(0)), dim=0)
        uav_rotate.append(uav_view_rotated)
    for k in range(1, 4):
        for i in range(25):
            uav_view_rotated = torch.rot90(uav_rotate[i], k=k, dims=(1, 2))
            uav_rotate.append(uav_view_rotated)

    print_time_flag = 0
    if print_time_flag:
        print("一次旋转时间", time.time() - part_time_start)
    show = 0
    if show:
        for i in range(100):
            show_np_pic(uav_rotate[i][0].cpu().numpy() * 255, "show_uav_and_goal_view", 1)  # 显示矩阵照片
            cv2.waitKey(0)
    return uav_rotate


def get_uav_rotate(uav_img, resize):
    uav_rotate = []
    uav_img_gray = cv2.cvtColor(uav_img, cv2.COLOR_RGB2GRAY)
    for i in range(25):
        uav_view_rotated = rrp_numpy(uav_img_gray, 3.6 * i, resize)
        uav_rotate.append(uav_view_rotated)

    for k in range(1, 4):
        for i in range(25):
            uav_view_rotated = np.rot90(uav_rotate[i], k=k, )
            uav_rotate.append(uav_view_rotated)
    return uav_rotate


def show_np_pic(array_in, name, resize):
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = array_in.shape[:2]
    scaled_image = cv2.resize(array_in.astype("uint8"), (int(width // resize), int(height // resize)),
                              interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, scaled_image)
