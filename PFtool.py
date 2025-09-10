import os
import cv2
import time
import math
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
import torch.nn.functional as ff
import matplotlib.pyplot as plt
import threading
from feature_extraction.extract import thread_match, thread_other
from feature_extraction.rrp import rrp, rrp_numpy, get_uav_rotate_g, get_uav_rotate



"""  数据path  数据文件路径展示
data
  -ligong
    -ligong_100m_1
      map
        img.png
        label.png
        detail.txt
      route
        PNG
          ligong_100m_1_000000.png
          ligong_100m_1_000200.png
          ...
      ligong_200m_1_idx1_all.csv                                                                       
"""


class ParticleState:
    # def __init__(self, data_path, prj_name, num_particles, height_max, height_min):
    def __init__(self, map_path, map_detail, particle_detail):
        np.random.seed(1)                   # 随机数【原本是能重新复现， 好像没什么用】
        self.cam_view = 2.0                 # 相机视野 限制2.0
        self.resize = 400                   # 图像缩放比例
        self.point_radius = 16              # 图像显示时的目标点半径
        self.points_mean = 0                # 粒子协方差
        self.print_time_flag = 0            # 是否打印输出一些值

        # 读取地图参数
        self.map_longitude_min = map_detail[0]
        self.map_longitude_max = map_detail[1]
        self.map_latitude_min = map_detail[2]
        self.map_latitude_max = map_detail[3]
        self.height_to_pix = map_detail[4]
        self.bei = map_detail[7]

        self.num_particles = particle_detail[0]     # 粒子初始化个数
        self.height_max = particle_detail[1]        # 限制最高高度
        self.height_min = particle_detail[2]        # 限制最低高度

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.map_og_path = map_path[0]                # 原始地图路径
        self.map_label_path = map_path[1]             # 语义地图路径
        assert os.path.exists(self.map_og_path), "file '{}' does not exist.".format(self.map_og_path)
        assert os.path.exists(self.map_label_path), "file '{}' does not exist.".format(self.map_label_path)

        self.map_label = np.asarray(Image.open(self.map_label_path), dtype=np.int8)                             # 语义地图
        self.map_label = np.where(self.map_label == 5, 1, self.map_label)                                       # 语义地图 将5(墙)转化为1(屋顶)
        self.map_height, self.map_width = self.map_label.shape[:2]                                              # 语义地图宽高
        self.pad = int(((self.height_max * self.height_to_pix * self.cam_view) // 500+1) * 500)                 # 边缘扩充距离 = 粒子max高度*self.cam_view*self.height_to_pix 按照500取整数
        self.map_label_pad = np.pad(self.map_label, pad_width=self.pad, mode='constant', constant_values=0)     # 扩充后的语义图

        self.map = cv2.imread(self.map_og_path)
        self.map_gray = cv2.cvtColor(self.map, cv2.COLOR_RGB2GRAY)
        self.map_gray_pad = np.pad(self.map_gray, pad_width=self.pad, mode='constant', constant_values=0)       # 扩充后的灰度地图

        wan1 = np.zeros((self.resize, self.resize))                                                             # 一个resize*resize的碗状矩阵
        wan1[self.resize//2:self.resize//2+1, self.resize//2:self.resize//2+1] = 1
        self.wan = np.clip(cv2.distanceTransform((255 - wan1 * 255).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE), a_min=None, a_max=255)+1
        self.wan_g = torch.from_numpy(self.wan).to(self.device)                                                 # 碗状矩阵传入GPU

        zero = np.zeros((self.map_height, self.map_width))
        zero_pad = np.pad(zero, pad_width=self.pad, mode='constant', constant_values=1)
        self.dist_p = cv2.distanceTransform(zero_pad.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)       # 一个跟扩充图相同的盆状矩阵
        self.dist_p_g = torch.from_numpy(cv2.distanceTransform(zero_pad.astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)).to(self.device)

        self.map_tags = dict()
        self.mapp_tags = dict()
        self.map_dist = dict()
        self.mapp_dist = dict()
        self.map_tags_g = dict()
        self.mapp_tags_g = dict()
        self.map_dist_g = dict()
        self.mapp_dist_g = dict()  # 【不知道怎么简写】
        self.map_tags['o'] = np.where((6 - self.map_label) != 6, 0, (6 - self.map_label)) // 6  # 代表非标准路面
        self.map_tags['b'] = np.where(self.map_label != 1, 0, self.map_label)                   # 代表建筑 tips：在前面已经将语义地图的5(墙)转化为1(屋顶)
        self.map_tags['r'] = np.where(self.map_label != 2, 0, self.map_label) // 2              # 代表标准马路
        self.map_tags['v'] = np.where(self.map_label != 3, 0, self.map_label) // 3              # 代表植被
        self.map_tags['w'] = np.where(self.map_label != 6, 0, self.map_label) // 6              # 代表水面
        self.map_tags['m'] = self.map_label                                                     # 代表原语义图
        """
        0 ['o'] other
        1 ['b'] wall/building
        2 ['r'] road
        3 ['v'] vehicle
        4       vehicle
        5       roof
        6 ['w'] water
        ['m'] 所有label
        """
        # 就诞生了：
        # map_tags      map_tags_g    地图语义图
        # mapp_tags     mapp_tags_g   地图扩充语义图
        # map_dist      map_dist_g    类别距离图
        # mapp_dist     mapp_dist_g   类别扩充距离图
        for key in self.map_tags:
            self.mapp_tags[key] = np.pad(self.map_tags[key], pad_width=self.pad, mode='constant', constant_values=0)
        for key in ['o', 'b', 'r', 'v', 'w']:
            self.map_dist[key] = np.clip(cv2.distanceTransform((255 - self.map_tags[key] * 255).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE), a_min=None, a_max=100)
        for key in self.map_dist:
            self.mapp_dist[key] = np.clip(cv2.distanceTransform((255 - self.mapp_tags[key] * 255).astype('uint8'), cv2.DIST_L2, cv2.DIST_MASK_PRECISE), a_min=None, a_max=100)
        for key in self.map_tags:
            self.map_tags_g[key] = torch.from_numpy(self.map_tags[key]).to(self.device)
            self.mapp_tags_g[key] = torch.from_numpy(self.mapp_tags[key]).to(self.device)
        for key in self.map_dist:
            self.map_dist_g[key] = torch.from_numpy(self.map_dist[key]).to(self.device)
            self.mapp_dist_g[key] = torch.from_numpy(self.mapp_dist[key]).to(self.device)

        # 以2*3的格式展现 方便后期检查
        self.map_tags_mix = np.block([[self.map_tags['o'], self.map_tags['b'], self.map_tags['r']], [self.map_tags['v'], self.map_tags['w'], self.map_tags['m']]])
        self.mapp_tags_mix = np.block([[self.mapp_tags['o'], self.mapp_tags['b'], self.mapp_tags['r']], [self.mapp_tags['v'], self.mapp_tags['w'], self.mapp_tags['m']]])
        self.map_dist_mix = np.block([[self.map_dist['o'], self.map_dist['b'], self.map_dist['r']], [self.map_dist['v'], self.map_dist['w'], np.zeros((self.map_height, self.map_width), dtype=int)]])
        self.mapp_dist_mix = np.block([[self.mapp_dist['o'], self.mapp_dist['b'], self.mapp_dist['r']], [self.mapp_dist['v'], self.mapp_dist['w'], np.zeros((self.map_height+2*self.pad, self.map_width+2*self.pad), dtype=int)]])

        self.particles = np.zeros((self.num_particles, 6)).astype(np.float32)

        init_method_mode = 0        # 分层随机方法  0:全层随机  1:分层随机  2:中心语义对齐随机
        if init_method_mode == 0:   # 全部随机
            for idx in range(self.num_particles):
                self.particles[idx, 0:4] = [random.random() for _ in range(4)]
            self.particles[:, 0] *= self.map_height
            self.particles[:, 1] *= self.map_width
            self.particles[:, 2] *= (self.height_max - self.height_min)
            self.particles[:, 2] += self.height_min
            self.particles[:, 3] *= 360  # theta
        elif init_method_mode == 1:  # 分层随机
            num_floor = 4
            for idx in range(self.num_particles//num_floor):
                self.particles[idx, 0:4] = [random.random() for _ in range(4)]
            step_num = self.num_particles//num_floor
            self.particles[:step_num, 0] *= self.map_height
            self.particles[:step_num, 1] *= self.map_width
            self.particles[:step_num, 2] = self.height_min
            self.particles[:step_num, 3] *= 360
            for m in range(1, num_floor):
                self.particles[step_num*m:step_num*(m+1), 0] = self.particles[:step_num, 0]
                self.particles[step_num*m:step_num*(m+1), 1] = self.particles[:step_num, 1]
                self.particles[step_num*m:step_num*(m+1), 2] = self.particles[:step_num, 2] + (self.height_max - self.height_min)//(num_floor-1) * m
                self.particles[step_num*m:step_num*(m+1), 3] = self.particles[:step_num, 3]
        elif init_method_mode == 2:  # 中心语义对齐随机
            for idx in range(self.num_particles):
                self.particles[idx, 0:4] = [random.random() for _ in range(4)]
                while True:
                    x, y, z, t = [random.random() for _ in range(4)]
                    if self.map_tags['b'][int(x * self.map_height), int(y * self.map_width)] == 1:
                        self.particles[idx, 0:4] = x, y, z, t
                        break
            self.particles[:, 0] *= self.map_height
            self.particles[:, 1] *= self.map_width

            self.particles[:, 2] *= (self.height_max - self.height_min)
            self.particles[:, 2] += self.height_min
            self.particles[:, 3] *= 360  # theta

        # 构建目标粒子矩阵
        self.particles_goal = np.zeros((2, 6)).astype(np.float32)
        print("初始化完成")

        self.img_idx = 0
        self.hang = 0
        self.pass_time = 0
        self.result = np.zeros((100, 6)).astype(np.float32)

    def init_particles_goal(self, latitude, longitude, height, theta):  # 初始化目标粒子

        aaa = self.gps2xyzt(latitude, longitude, height, theta)
        self.particles_goal[0, 0], self.particles_goal[0, 1], self.particles_goal[0, 2], self.particles_goal[0, 3] = self.gps2xyzt(latitude, longitude, height, theta)

    def init_particles_goal_set(self, idx, set_x, set_y, set_z, set_theta):  # 初始化目标粒子
        # 直接定义粒子位置
        self.particles_goal[idx, 0] = set_x
        self.particles_goal[idx, 1] = set_y
        self.particles_goal[idx, 2] = set_z
        self.particles_goal[idx, 3] = set_theta

    def update_goal_1(self, latitude, longitude, height, theta):  # 迭代目标粒子 方法1：直接位移
        # 根据目标 粒子经纬度 和图像四周经纬度定位目标粒子在图像中的位置
        self.particles_goal[0, 0] = self.map_height * (self.map_latitude_max - latitude) / (self.map_latitude_max - self.map_latitude_min)
        self.particles_goal[0, 1] = self.map_width * (longitude - self.map_longitude_min) / (self.map_longitude_max - self.map_longitude_min)
        self.particles_goal[0, 2] = height
        self.particles_goal[0, 3] = theta

    def update_goal(self, x_move, y_move, z_move, t_move):  # 迭代目标粒子 方法2：直接位移
        # 根据传感器的输入进行位移
        self.particles_goal[0, 0] = self.particles_goal[0, 0] + x_move
        self.particles_goal[0, 1] = self.particles_goal[0, 1] + y_move
        self.particles_goal[0, 2] = self.particles_goal[0, 2] + z_move
        self.particles_goal[0, 3] = self.particles_goal[0, 3] + t_move
        while self.particles_goal[0, 3] >= 360:
            self.particles_goal[0, 3] -= 360
        while self.particles_goal[0, 3] < 0:
            self.particles_goal[0, 3] += 360

    def show_goal_view(self, img, resize):  # 输入无人机语义图
        particle_angle = -self.particles_goal[0, 3]                              # 目标粒子角度
        uav_view_og_gpu = torch.from_numpy(img).to(self.device)                  # 无人机语义图转入GOU
        rotated_uav_view_og = rrp(uav_view_og_gpu, particle_angle, self.resize)  # 无人机旋转矩阵

        tx, ty = round(self.particles_goal[0, 0].item()), round(self.particles_goal[0, 1].item())           # 目标粒子平移量
        cut_size = round(((self.particles_goal[0, 2].item()) * self.cam_view * self.height_to_pix) / 2)     # 裁剪的像素
        map_label_cut = self.map_label_pad[tx + self.pad - cut_size:tx + self.pad + cut_size, ty + self.pad - cut_size:ty + self.pad + cut_size]  # 地图语义矩阵裁剪
        map_label_cut_resize = cv2.resize(map_label_cut, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)  # 地图语义矩阵裁剪resize
        a = np.hstack((rotated_uav_view_og.squeeze(0).cpu().numpy(), map_label_cut_resize+1))       # 无人机图和地图拼接
        show_np_pic(a * 30, "show_uav_and_goal_view", resize)                                 # 显示矩阵照片
        cv2.waitKey(0)

    def update_particles(self, uav_img):
        uav_rotate = get_uav_rotate_g(uav_img, self.device, self.resize)
        part_time_start = time.time()
        # self.particles[0, :] = self.particles_goal[0, :]  # zuobi
        # self.particles[:, 3] = self.particles_goal[0, 3]  # zuobi
        for particle_idx in range(self.particles.shape[0]):
            if self.particles[particle_idx, 0] <= 0 or self.particles[particle_idx, 1] <= 0 or self.particles[particle_idx, 0] >= self.map_height or self.particles[particle_idx, 1] >= self.map_width:
                self.particles[particle_idx, 4] = 0
            else:
                tx, ty = round(self.particles[particle_idx, 0].item()), round(self.particles[particle_idx, 1].item())  # 单粒子平移量
                cut_size = round(((self.particles[particle_idx, 2].item()) * self.cam_view * self.height_to_pix) / 2)
                cut_x_min = tx + self.pad - cut_size
                cut_x_max = tx + self.pad + cut_size
                cut_y_min = ty + self.pad - cut_size
                cut_y_max = ty + self.pad + cut_size
                map0 = self.mapp_dist_g['o'][cut_x_min:cut_x_max, cut_y_min:cut_y_max]
                map1 = self.mapp_dist_g['b'][cut_x_min:cut_x_max, cut_y_min:cut_y_max]
                map2 = self.mapp_dist_g['r'][cut_x_min:cut_x_max, cut_y_min:cut_y_max]
                map3 = self.mapp_dist_g['v'][cut_x_min:cut_x_max, cut_y_min:cut_y_max]
                # result2 = torch.cat((map0.unsqueeze(0), map1.unsqueeze(0), map2.unsqueeze(0)), dim=0)  #
                result2 = torch.cat((map0.unsqueeze(0), map1.unsqueeze(0), map2.unsqueeze(0), map3.unsqueeze(0)), dim=0)

                result2 = ff.interpolate(result2.unsqueeze(0), size=(self.resize, self.resize), mode='nearest') * self.wan_g  #  * self.particles[particle_idx, 2]
                weight_sum = torch.sum(uav_rotate[int((360 - self.particles[particle_idx][3] + 1.8) / 3.6) % 100] * result2.squeeze(0), dim=(0, 1, 2)) + 10 + 100000 * torch.sum(self.dist_p_g[cut_x_min:cut_x_max, cut_y_min:cut_y_max])
                self.particles[particle_idx, 4] = self.resize * self.resize / (weight_sum * weight_sum)
        if self.print_time_flag:
            use_time = time.time() - part_time_start
            print("一次迭代时间", time.time() - part_time_start)
            self.pass_time = use_time
        self.remap()
    def remap(self):  # 粒子重映射
        for particle_idx in range(self.num_particles):
            if particle_idx == 0:
                self.particles[particle_idx, 5] = self.particles[particle_idx, 4]  # particles[idx, 4]单个粒子权重  particles[idx, 5]idx以上所有粒子权重和
            else:
                self.particles[particle_idx, 5] = self.particles[particle_idx, 4] + self.particles[particle_idx - 1, 5]
        max_value = self.particles[self.num_particles-1, 5]
        self.particles[:, 5] = self.particles[:, 5] / max_value
        new_num_particles = round(4 / 5 * self.num_particles)           # 迭代后的粒子个数
        if new_num_particles < 1000:
            new_num_particles = 1000
        x = np.random.rand(new_num_particles)                           # 随机生成new_num_particles个数
        indexes = np.searchsorted(self.particles[:, 5], x)              # 根据生成的数对应idx
        self.particles[:new_num_particles] = self.particles[indexes]    # 将particles矩阵重新提取
        self.particles = self.particles[:new_num_particles, :]          # 将particles矩阵缩短
        self.num_particles = new_num_particles                          # 更新粒子个数

    def move(self, latitude, longitude, height, t_now, t_old):
        delta_latitude = self.map_height * latitude / (self.map_latitude_max - self.map_latitude_min)
        delta_longitude = self.map_width * longitude / (self.map_longitude_max - self.map_longitude_min)
        delta_height = height
        if delta_latitude == 0:
            if delta_longitude > 0:
                theta = 90
            elif delta_longitude < 0:
                theta = 270
            else:
                theta = 0
        else:
            if delta_latitude > 0:
                theta = math.atan(delta_longitude / delta_latitude) / math.pi * 180  # math.atan输出为弧度制  math.sin也需要弧度制
            else:
                theta = math.atan(delta_longitude / delta_latitude) / math.pi * 180 + 180
        length = math.sqrt(delta_longitude * delta_longitude + delta_latitude * delta_latitude)
        theta1 = theta - t_old
        theta2 = t_now - theta

        for particle_idx in range(self.num_particles):
            self.particles[particle_idx, 0] = self.particles[particle_idx, 0] - length * math.cos((theta1 + self.particles[particle_idx, 3]) / 180 * math.pi)  + random.gauss(2, 15)
            self.particles[particle_idx, 1] = self.particles[particle_idx, 1] + length * math.sin((theta1 + self.particles[particle_idx, 3]) / 180 * math.pi)  + random.gauss(2, 15)
            if self.height_min != self.height_max:
                self.particles[particle_idx, 2] = self.particles[particle_idx, 2] + delta_height + random.gauss(2, 8)
            self.particles[particle_idx, 3] = self.particles[particle_idx, 3] + theta1 + theta2 + random.gauss(0, 0.2)

            while self.particles[particle_idx, 3] >= 360:
                self.particles[particle_idx, 3] -= 360
            while self.particles[particle_idx, 3] < 0:
                self.particles[particle_idx, 3] += 360
            if self.particles[particle_idx, 2] < self.height_min:
                self.particles[particle_idx, 2] = self.height_min
            if self.particles[particle_idx, 2] > self.height_max:
                self.particles[particle_idx, 2] = self.height_max
            if self.particles[particle_idx, 0] <= 0 or self.particles[particle_idx, 1] <= 0 or self.particles[particle_idx, 0] >= self.map_height or self.particles[particle_idx, 1] >= self.map_width:
                self.particles[particle_idx, :] = 0
        # self.center()


    def show_pic(self, obj, save_path, resize, wait):
        if obj == "map_tags_mix":                                                   # 2*3分类显示地图语义图
            show_np_pic(self.map_tags_mix * 250, 'map_tags_mix', resize)
        elif obj == "mapp_tags_mix":                                                # 2*3分类显示地图扩充语义图
            show_np_pic(self.mapp_tags_mix * 250, 'mapp_tags_mix', resize)
        elif obj == "map_dist_mix":                                                 # 2*3分类显示类别距离图
            show_np_pic(self.map_dist_mix * 2, 'map_dist_mix', resize)
        elif obj == "mapp_dist_mix":                                                # 2*3分类显示类别扩充距离图
            show_np_pic(self.mapp_dist_mix * 2, 'mapp_dist_mix', resize)
        elif obj == "dist_p_g":                                                     # 显示盆状矩阵
            show_np_pic(self.dist_p, 'dist_p_g', resize)
        elif obj == "map_tag":                                                      # 显示地图语义图
            show_np_pic(self.map_tags['m']*40, 'map_tags', resize)
        elif obj == "map_og":
            background_path = cv2.imread(self.map_og_path)
            height_background, width_background = background_path.shape[:2]
            scaled_image = cv2.resize(background_path, (width_background // resize, height_background // resize), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("map_og", scaled_image)
        elif obj == "point" or obj == "goal" or obj == "point_goal":
            img_path = self.map_og_path
            background_path = cv2.imread(img_path)
            height_background, width_background = background_path.shape[:2]

            height_array, width_array = self.particles.shape[:2]

            color1 = (0, 0, 255)  # 红色
            color2 = (0, 255, 0)  # 绿色
            radius = self.point_radius
            if obj == "point":
                for idx in range(height_array):
                    cv2.circle(background_path, (int(self.particles[idx, 1]), int(self.particles[idx, 0])), radius//4, color2, -1)
            elif obj == "goal":
                cv2.circle(background_path, (int(self.particles_goal[0, 1]), int(self.particles_goal[0, 0])), radius, color1, -1)
            elif obj == "point_goal":

                for idx in range(height_array):
                    cv2.circle(background_path, (int(self.particles[idx, 1]), int(self.particles[idx, 0])), radius // 4, color2, -1)
                cv2.circle(background_path, (int(self.particles_goal[0, 1]), int(self.particles_goal[0, 0])), radius, color1, -1)
            scaled_image = cv2.resize(background_path, (width_background // resize, height_background // resize), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("Img show", scaled_image)
            if save_path != "":
                cv2.imwrite(save_path, scaled_image)
        # cv2.waitKey(wait)

    def save_csv(self, csv_save_path):
        matrix = np.vstack((self.particles, self.particles_goal))
        df = pd.DataFrame(matrix)
        df.to_csv(csv_save_path, index=False, header=False)

    def save_csv2(self, csv_save_path):
        # matrix = np.vstack((self.result, self.particles_goal))
        df = pd.DataFrame(self.result)
        df.to_csv(csv_save_path, index=False, header=False)


    def show3D(self,):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 在3D子图上绘制散点图
        ax.scatter(self.particles[:, 0], self.particles[:, 1], self.particles[:, 2])
        # 设置坐标轴标签
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim([0, self.map_height])
        ax.set_ylim([0, self.map_width])
        ax.set_zlim([0, self.height_max])
        # 显示图形
        plt.show()

    def center(self):
        _particles = self.particles.copy()
        x_mean = np.sum(_particles[:, 0]) / _particles.shape[0]
        y_mean = np.sum(_particles[:, 1]) / _particles.shape[0]
        z_mean = np.sum(_particles[:, 2]) / _particles.shape[0]
        _particles[:, 0] = _particles[:, 0] - x_mean
        _particles[:, 1] = _particles[:, 1] - y_mean
        _particles[:, 2] = _particles[:, 2] - z_mean
        _particles[:, 3] = np.sqrt(_particles[:, 0] * _particles[:, 0] + _particles[:, 1] * _particles[:, 1])
        column_sum = np.sum(_particles[:, 3])
        print("协方差:", column_sum)

        self.result[self.hang, 0] = self.img_idx
        self.result[self.hang, 1] = self.pass_time
        self.result[self.hang, 2] = 0
        self.result[self.hang, 3] = 0
        self.result[self.hang, 4] = 0
        self.result[self.hang, 5] = 0


        if self.points_mean == 0 and column_sum > 0:
            self.points_mean = column_sum
        if column_sum < self.points_mean / 10:
            points = self.particles  # 所有点
            # 使用DBSCAN进行密度聚类
            dbscan = DBSCAN(eps=30, min_samples=20)  # 区域半径为30 最小个数为20
            labels = dbscan.fit_predict(points)  # 这里已经预测出点的类别
            # 获取所有非离群点的索引
            valid_indices = labels != -1
            filtered_points = points[valid_indices]  # 有效点
            filtered_labels = labels[valid_indices]  # 有效标签

            # 计算每个簇的中心
            clusters = {}
            for idx, label in enumerate(filtered_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(filtered_points[idx])

            centers = {}
            for label, cluster_points in clusters.items():
                cluster_array = np.array(cluster_points)
                centers[label] = cluster_array.mean(axis=0)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # 在3D子图上绘制散点图
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.5)
            outlier_points = points[labels == -1]
            ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2], c='red', alpha=0.8, label='Outliers')

            colors = ['blue', 'green', 'orange', 'purple']  # 支持最多4个簇的显示

            best_x, best_y, best_z, best_w = 0, 0, 0, 10000

            if len(centers) <= 4:
                for label, center in centers.items():
                    ax.scatter(center[0], center[1], center[2], c=colors[label],
                                marker='*', edgecolor='black',
                                label=f'Cluster {label} Center')
                    print("x误差", center[0] - self.particles_goal[0, 0], "y误差", center[1] - self.particles_goal[0, 1], "z误差", center[2] - self.particles_goal[0, 2], "定位误差：", math.sqrt((center[0] - self.particles_goal[0, 0]) ** 2 + (center[1] - self.particles_goal[0, 1]) ** 2))
                    if math.sqrt((center[0] - self.particles_goal[0, 0]) ** 2 + (center[1] - self.particles_goal[0, 1]) ** 2) < best_w:
                        best_x = center[0] - self.particles_goal[0, 0]
                        best_y = center[1] - self.particles_goal[0, 1]
                        best_z = center[2] - self.particles_goal[0, 2]
                        best_w = math.sqrt((center[0] - self.particles_goal[0, 0]) ** 2 + (center[1] - self.particles_goal[0, 1]) ** 2)
                self.result[self.hang, 2] = best_x
                self.result[self.hang, 3] = best_y
                self.result[self.hang, 4] = best_z
                self.result[self.hang, 5] = best_w


            # 设置坐标轴标签
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim([0, self.map_height])
            ax.set_ylim([0, self.map_width])
            ax.set_zlim([0, self.height_max])
            # # 显示图形
            # plt.show()
        self.hang += 1

    def gps2xyzt(self, latitude, longitude, height, theta):
        # return (30.315494 - latitude) * self.bei + self.x_start, (longitude - 120.342894) * self.bei + self.y_start, height, theta
        return self.map_height * (self.map_latitude_max - latitude) / (self.map_latitude_max - self.map_latitude_min), self.map_width * (longitude - self.map_longitude_min) / (self.map_longitude_max - self.map_longitude_min), height, theta

def show_np_pic(array_in, name, resize):
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    height, width = array_in.shape[:2]
    scaled_image = cv2.resize(array_in.astype("uint8"), (int(width // resize), int(height // resize)),
                              interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, scaled_image)
