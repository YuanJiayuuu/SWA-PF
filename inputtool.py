import os
import cv2
import pandas as pd

class img_input:
    def __init__(self, data_path, prj_name):
        self.bei = 390000
        self.x_start = 1220
        self.y_start = 860

        data_path = os.path.join(data_path, prj_name)                   # 读取数据路径"E:/MAFS_DATA/MAFS//" 路径名称"dianli_bian_1"
        txt_path = os.path.join(data_path, 'route', 'all.txt')          # 读取图片名称集合
        csv_path = os.path.join(data_path, prj_name + '_idx1_all.csv')  # 读取经纬高数据集合
        air_route_png_path = os.path.join(data_path, 'route', 'PNG')    # 图片数据路径
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]   # 图片名称集合
        self.route_idx = [int(file_name[-6:]) for file_name in file_names]          # 图片id集合
        self.images = [os.path.join(air_route_png_path, file_name) + ".png" for file_name in file_names]  # 图片路径原名称集合
        for image in self.images:
            assert os.path.exists(image), "file '{}' does not exist.".format(image)
        assert os.path.exists(csv_path), "file '{}' does not exist.".format(csv_path)
        self.move_array = pd.read_csv(csv_path, header=None).to_numpy()     # 经纬高数据集合
        self.num_route = len(self.images)                                   # 路径照片长度

    # 根据idx返回图像数据
    def back_img(self, index):
        uav = cv2.imread(self.images[index], 1)
        return uav

    # 根据idx返回目标像素位置
    def back_move(self, index):
        longitude, latitude, height, yaw = self.move_array[self.route_idx[index], 1].item(), self.move_array[self.route_idx[index], 2].item(), self.move_array[self.route_idx[index], 3].item(), self.move_array[self.route_idx[index], 4].item()
        return (30.315494 - latitude) * self.bei + self.x_start, (longitude - 120.342894) * self.bei + self.y_start, height, yaw

    # 根据idx返回目标真实经纬度位置
    def back_all(self, index):
        longitude, latitude, height, yaw = self.move_array[self.route_idx[index], 1].item(), self.move_array[self.route_idx[index], 2].item(), self.move_array[self.route_idx[index], 3].item(), self.move_array[self.route_idx[index], 4].item()
        return latitude, longitude, height, yaw
