<h1 align="center"> SWA-PF: Semantic-Weighted Adaptive Particle Filter for Memory-Efficient 4-DoF UAV Localization in GNSS-Denied Environments </h1>

We shared our results [here](【视觉无人机定位】 https://www.bilibili.com/video/BV1bzQJYzEnA/?share_source=copy_web&vd_source=fa7d4ca9e5eb2082828642578bec4de4)

![](https://github.com/YuanJiayuuu/SWA-PF/blob/main/pic/IMG1.png)

# About Dataset

MAFS

```
├── MAFS/
│   ├──ligong/
│       ├── 200m
│           ├── map
│               ├── detail.txt				/* details about the satellite image
│               ├── img.png					/* satellite image
│               ├── label.png				/* Semantic satellite map
│           ├── route
│               ├── gps.txt					/* format as: path latitude longitude height
│               ├── ligong_200m_1_000000.png	/* UAV images
│               ├── ligong_200m_1_000050.png
│               ...
│   ├──hangdian/
│       ├── 150m
│           ├── map
│           ├── route
│               ├── gps.txt
│               ├── ligong_200m_1_000000.png
│               ├── ligong_200m_1_000050.png
│               ...
│       ├── 200m
│   ...
```

## Prerequisites

- Python 3.9+
- GPU Memory >= 8G
- Numpy = 1.26
- Pytorch 1.10.0+cu113
- Torchvision 0.11.1+cu113

## Installation

It is best to use cuda version 11.3 and pytorch version 1.10.0. 

You can execute the following command to install all dependencies.

```
conda create -n SWAPF python=3.9
pip install "torch-1.10.0+cu113-cp39-cp39-win_amd64.whl"
pip install "torchvision-0.11.1+cu113-cp39-cp39-win_amd64.whl"
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install timm
pip install einops
pip uninstall numpy
pip install numpy==1.26
```

## Dataset & Preparation

You can download the MAFS datasets from this [website](

## Start

You can download the corresponding version from this [website](

## Citation

The following paper uses and reports the result of the baseline model. You may cite it in your paper.

```bibtex
@ARTICLE{DenseUAV,
  author={Dai, Ming and Zheng, Enhui and Feng, Zhenhua and Qi, Lei and Zhuang, Jiedong and Yang, Wankou},
  journal={IEEE Transactions on Image Processing},
  title={Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments},
  year={2024},
  volume={33},
  number={},
  pages={493-508},
  doi={10.1109/TIP.2023.3346279}}
```

## Related Work

- University-1652 [https://github.com/layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline)
- FSRA [https://github.com/Dmmm1997/FSRA](
