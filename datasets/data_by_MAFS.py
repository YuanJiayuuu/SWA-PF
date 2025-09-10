import os
import sys
from torch.utils.data import Dataset, DataLoader


class MAFS_output:
    def __init__(self, path):
        if not os.path.exists(path):
            sys.exit(f"错误：路径 '{path}' 错误。")
        print(f"正在运行路径 ：'{path}' ")
        route_path = os.path.join(path, 'route')
        self.map_path = os.path.join(path, 'map')
        gps_path = os.path.join(route_path, 'gps.txt')

        with open(os.path.join(gps_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.img_dict = []
        self.position = []

        for file_name in file_names:
            parts = file_name.split()
            # print(os.path.join(route_path, parts[0]))
            self.img_dict.append(os.path.join(route_path, parts[0]))
            self.position.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    # def __getitem__(self, index):
    #     img = self.img_dict[index]
    #     pos = self.position[index]
    #     return img, pos

    def __getitem__(self, index):
        img_path = self.img_dict[index]
        pos = self.position[index]
        return img_path, pos

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_map(self):
        image_path = os.path.join(self.map_path, 'img.png')
        label_path = os.path.join(self.map_path, 'label.png')
        detail_path = os.path.join(self.map_path, 'detail.txt')
        with open(detail_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()  # 移除当前行的首尾空白字符。
                if 'longitude_min' in line:
                    position = line.find("longitude_min")
                    lon_min = float(line[position + 13:])
                elif 'longitude_max' in line:
                    position = line.find("longitude_max")
                    lon_max = float(line[position + 13:])
                elif 'latitude_min' in line:
                    position = line.find("latitude_min")
                    lat_min = float(line[position + 12:])
                elif 'latitude_max' in line:
                    position = line.find("latitude_max")
                    lat_max = float(line[position + 12:])
                elif 'height_to_pix' in line:
                    position = line.find("height_to_pix")
                    height_to_pix = float(line[position + 13:])
                elif 'jw_to_pix_x' in line:
                    position = line.find("jw_to_pix_x")
                    jw_to_pix_x = int(line[position + 12:])
                elif 'jw_to_pix_y' in line:
                    position = line.find("jw_to_pix_y")
                    jw_to_pix_y = int(line[position + 12:])
                elif 'bei' in line:
                    position = line.find("bei")
                    bei = int(line[position + 4:])
                # elif 'x_start' in line:
                #     position = line.find("x_start")
                #     x_start = int(line[position + 7:])
                # elif 'y_start' in line:
                #     position = line.find("y_start")
                #     y_start = int(line[position + 7:])
        return [image_path, label_path], [lon_min, lon_max, lat_min, lat_max, height_to_pix, jw_to_pix_x, jw_to_pix_y, bei]

    def __len__(self):
        return len(self.img_dict)
