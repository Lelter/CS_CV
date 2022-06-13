import sys

import cv2
import numpy as np


def img_ssd(templateImg, targetImg):
    # 模板匹配SSD
    # 参数：模板图，目标图
    # 返回：目标图的框
    M = targetImg.shape[0]
    m = templateImg.shape[0]
    N = targetImg.shape[1]
    n = templateImg.shape[1]
    Range_x = M - m - 1
    Range_y = N - n - 1
    src_img = cv2.cvtColor(targetImg, cv2.COLOR_RGB2GRAY)
    search_img = cv2.cvtColor(templateImg, cv2.COLOR_RGB2GRAY)
    min_res = sys.maxsize
    Best_x = 0
    Best_y = 0
    for i in range(Range_x):
        for j in range(Range_y):
            subgraph_img = src_img[i:i + m, j:j + n]
            res = np.sum((search_img.astype("float") - subgraph_img.astype("float")) ** 2)  # SSD公式
            if res < min_res:
                min_res = res
                Best_x = i
                Best_y = j
    return Best_y, Best_x
    pass
