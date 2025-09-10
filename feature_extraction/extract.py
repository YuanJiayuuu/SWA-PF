import cv2
import numpy as np
from sklearn.metrics import mutual_info_score


def thread_match(self, num, max_row, descriptors, update_method, method, bf):
    for i in range(300):
        particle_idx = num * 300 + i
        if particle_idx < max_row:
            if self.particles[particle_idx, 0] <= 0 or self.particles[particle_idx, 1] <= 0 or self.particles[particle_idx, 0] >= self.map_height or self.particles[particle_idx, 1] >= self.map_width:
                self.particles[particle_idx, :] = 0
                continue

            tx, ty = round(self.particles[particle_idx, 0].item()), round(self.particles[particle_idx, 1].item())  # 目标粒子平移量
            cut_size = round(((self.particles[particle_idx, 2].item()) * self.cam_view * self.height_to_pix) / 2)
            map_label_cut = self.map_gray_pad[tx + self.pad - cut_size:tx + self.pad + cut_size, ty + self.pad - cut_size:ty + self.pad + cut_size]  # 地图语义矩阵裁剪
            map_label_cut_resize = cv2.resize(map_label_cut, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)  # 地图语义矩阵裁剪resize
            descriptors1 = descriptors[int((360 - self.particles[particle_idx][3] + 1.8) / 3.6) % 100]
            _, descriptors2 = method.detectAndCompute(map_label_cut_resize, None)
            similarity = 0.0
            if update_method == "ORB":
                matches = bf.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
                good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
                similarity = len(good) / len(matches)
            elif update_method == "SIFT":
                # 使用FLANN匹配器（更适合SIFT）
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                if len(matches) > 0:
                    similarity = len(good) / len(matches)
                else:
                    similarity = 0.0  # 处理无匹配的情况
            elif update_method == "AKAZE":
                matches = bf.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
                good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
                similarity = len(good) / len(matches)
            elif update_method == "BRISK":
                matches = bf.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
                good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
                similarity = len(good) / len(matches)
            elif update_method == "KAZE":
                matches = bf.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)
                good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
                similarity = len(good) / len(matches)

            self.particles[particle_idx, 4] = similarity * similarity


def thread_other(self, num, max_row, descriptors, update_method):
    for i in range(300):
        particle_idx = num * 300 + i
        if particle_idx < max_row:
            if self.particles[particle_idx, 0] <= 0 or self.particles[particle_idx, 1] <= 0 or self.particles[particle_idx, 0] >= self.map_height or self.particles[particle_idx, 1] >= self.map_width:
                self.particles[particle_idx, :] = 0
                continue

            tx, ty = round(self.particles[particle_idx, 0].item()), round(self.particles[particle_idx, 1].item())  # 目标粒子平移量
            cut_size = round(((self.particles[particle_idx, 2].item()) * self.cam_view * self.height_to_pix) / 2)
            map_label_cut = self.map_gray_pad[tx + self.pad - cut_size:tx + self.pad + cut_size, ty + self.pad - cut_size:ty + self.pad + cut_size]  # 地图语义矩阵裁剪
            map_label_cut_resize = cv2.resize(map_label_cut, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)  # 地图语义矩阵裁剪resize
            descriptors1 = descriptors[int((360 - self.particles[particle_idx][3] + 1.8) / 3.6) % 100]
            similarity = 0.0
            if update_method == "MI":
                descriptors2 = map_label_cut_resize.flatten()
                mi = mutual_info_score(descriptors1, descriptors2)
                max_mi = np.log2(len(descriptors1))  # 最大可能的互信息值
                similarity = mi / max_mi if max_mi > 0 else 0.0
            elif update_method == "calcHist":
                descriptors2 = map_label_cut_resize.flatten()
                hist_2d, _, _ = np.histogram2d(descriptors1, descriptors2, bins=32)

                # 计算概率分布
                eps = 1e-10
                P_xy = hist_2d / np.sum(hist_2d)
                P_x = np.sum(P_xy, axis=1)
                P_y = np.sum(P_xy, axis=0)

                # 计算熵
                H_x = -np.sum(P_x * np.log2(P_x + eps))
                H_y = -np.sum(P_y * np.log2(P_y + eps))
                H_xy = -np.sum(P_xy * np.log2(P_xy + eps))

                # 计算互信息
                MI = H_x + H_y - H_xy

                # 归一化互信息
                if H_x + H_y < eps:
                    similarity = 1.0
                else:
                    similarity = (2 * MI) / (H_x + H_y)
            self.particles[particle_idx, 4] = similarity * similarity
