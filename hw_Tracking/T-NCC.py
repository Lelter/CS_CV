import sys

import cv2
import numpy as np


def img_ncc(templateImg, targetImg):
    # 模板匹配NCC
    # 参数：模板图，目标图
    # 返回：最佳坐标
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


    return Best_x, Best_y
    pass