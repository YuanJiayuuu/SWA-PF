import os
import sys

from datasets.data_by_MAFS import MAFS_output
from torch.utils.data import DataLoader

def data_output(name, idx, height):
    if name == 'MAFS':
        data_path = 'D:/DATASETS/MAFS'                                   # 数据集绝对路径
        route_path = os.path.join(data_path, str(idx).zfill(2), height)  # 某一段路径
        out = MAFS_output(route_path)
        # out = DataLoader(dataset, batch_size=1, shuffle=False)
        return out
    # elif name == 'DenseUAV':
    #     out = DenseUAV_output(name, idx)
    #     pass
    else:
        raise SystemExit("Datasets ERROR")
        pass






