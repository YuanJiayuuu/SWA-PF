import os
import cv2
import PFtool
import datetime
import inputtool
import torch
from PIL import Image
import numpy as np
from segformer import SegFormer_Segmentation
from feature_extraction.rrp import rrp_DL
import torch.nn.functional as ff
from datasets.data_process import data_output
import time


def swa_pf():
    # 1选择数据集
    # 2选择方法
    # 开始定位

    dataset = 'MAFS'  # 目前支持MAFS
    route = 'hangdian'
    height_str = '150m'
    update_method = 'SWAPF'
    particle_detail = [20000, 150, 150]
    data_route = data_output(dataset, route, height_str)
    map_path, map_detail = data_route.get_map()
    print(map_path, map_detail)
    segformer = SegFormer_Segmentation()
    # 传入参数【地图路径map_path、地图参数map_detail、粒子[个数、高度]】
    particles = PFtool.ParticleState(map_path, map_detail, particle_detail)  # 初始化粒子参数
    mode = 0

    latitude, longitude, height, theta = 0, 0, 0, 0
    old_latitude, old_longitude, old_height, old_theta = 0, 0, 0, 0

    if mode == 0:
        index = 0
        save_path, csv_save_path = make_save_path(route, height_str)
        for img_path, pos in data_route:
            print(f"正在读取", img_path)
            uav_image = cv2.imread(img_path, 1)

            # 步骤1: 读取初始数据
            if index == 0:
                longitude, latitude, height, theta = pos
                old_longitude, old_latitude, old_height, old_theta = pos
                particles.init_particles_goal(latitude, longitude, height, theta)
            else:
                longitude, latitude, height, theta = pos
                particles.update_goal_1(latitude, longitude, height, theta)
                particles.move(latitude - old_latitude, longitude - old_longitude, height - old_height, theta, old_theta)
                old_latitude, old_longitude, old_height, old_theta = latitude, longitude, height, theta

            # 有图片就能迭代
            # 步骤2：更新粒子
            if update_method == 'SWAPF':
                image = uav_image[:, :, ::-1]  # BGR转RGB
                image = Image.fromarray(image)  # 转化为Image格式
                image, pr = segformer.detect_image(image, count=False, name_classes=7)  # 生成语义图
                pr = np.where(pr == 5, 1, pr)  # 将屋顶变为墙
                particles.update_particles(pr)  # 根据语义图迭代粒子

            # # 步骤3: 保存CSV
            # particles.save_csv(csv_save_path)
            #
            # # 步骤4: 重映射粒子
            particles.remap()
            #
            # 步骤5: 保存图片
            png_save_path = os.path.join(save_path, str(index) + ".png")
            particles.show_pic(obj="point_goal", save_path=png_save_path, resize=5, wait=1)

            # # 步骤6: 粒子居中
            # # particles.center()
            #
            # # 步骤8: 保存结果CSV
            # particles.save_csv2(csv_save_path)
            # # if route_index > 40:
            # #     particles.show3D()

            index += 1


    # # 2 确定self.height_to_pix(高度和像素的裁剪比例)
    # elif mode == 2:
    #     for route_index in range(img_input.num_route - 1):
    #         latitude, longitude, height, theta = img_input.back_all(route_index)
    #         particles.update_goal_1(latitude, longitude, height, theta)
    #
    #         uav_image = img_input.back_img(route_index)
    #         uav_img_gray = cv2.cvtColor(uav_image, cv2.COLOR_RGB2GRAY)
    #         uav_rotate = PFtool.rrp_numpy(uav_img_gray, -particles.particles_goal[0, 3].item(), 400)
    #
    #         tx, ty = round(particles.particles_goal[0, 0].item()), round(particles.particles_goal[0, 1].item())  # 单粒子平移量
    #         cut_size = round(((particles.particles_goal[0, 2].item()) * particles.cam_view * 0.95) / 2)
    #         map_gray_cut = particles.map_gray_pad[tx + particles.pad - cut_size:tx + particles.pad + cut_size, ty + particles.pad - cut_size:ty + particles.pad + cut_size]  # 地图语义矩阵裁剪
    #         map_gray_cut = cv2.resize(map_gray_cut, (400, 400))
    #
    #         cv2.imshow("uav", uav_rotate)
    #         cv2.imshow("map", map_gray_cut)
    #         cv2.waitKey(0)


def make_save_path(route, height):
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M%S_')
    save_path = os.path.join("./save_result", time_str + route + height)
    os.makedirs(save_path)
    csv_save_path = os.path.join(save_path, route + '_' + height + ".csv")
    return save_path, csv_save_path

if __name__ == '__main__':
    swa_pf()
