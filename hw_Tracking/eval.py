import cv2
import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    g = np.asarray(g)
    p1 = [int(p[0]), int(p[1]) + int(p[3]), int(p[0]), int(p[1]), int(p[0]) + int(p[2]), int(p[1]),
          int(p[0]) + int(p[2]), int(p[1]) + int(p[3])]
    p = np.asarray(p1)

    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def evalMatch(templateRecList, imgRecList, width=0, height=0, function='LK'):
    # 参数：templateRecList：匹配框的坐标，imgRecList：真实框的坐标列表
    # 跟对的帧数列表
    # TargetRec为(左上角x,y,宽，高)
    # templateRec为(左上角x,y)
    if function == 'ssd' or function == 'ncc':
        width = int(imgRecList[0][2])
        height = int(imgRecList[0][3])
        rightList = []
        for i in range(len(imgRecList)):
            px1, py1, px2, py2 = int(templateRecList[i][0]), int(templateRecList[i][1]), int(
                templateRecList[i][0]) + width, \
                                 int(templateRecList[i][1]) + height
            # print("预测框P的坐标是：({}, {}, {}, {})".format(px1, py1, px2, py2))

            gx1, gy1, gx2, gy2 = int(imgRecList[i][0]), int(imgRecList[i][1]), int(imgRecList[i][0]) + int(
                imgRecList[i][2]), int(imgRecList[i][1]) + int(imgRecList[i][3])
            # print("原标记框G的坐标是：({}, {}, {}, {})".format(gx1, gy1, gx2, gy2))

            parea = (px2 - px1) * (py2 - py1)  # 计算P的面积
            garea = (gx2 - gx1) * (gy2 - gy1)  # 计算G的面积
            # print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

            # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
            x1 = max(px1, gx1)  # 得到左上顶点的横坐标
            y1 = max(py1, gy1)  # 得到左上顶点的纵坐标
            x2 = min(px2, gx2)  # 得到右下顶点的横坐标
            y2 = min(py2, gy2)  # 得到右下顶点的纵坐标

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                rightList.append(i)
                continue

            area = w * h  # G∩P的面积
            # print("G∩P的面积是：{}".format(area))

            # 并集的面积 = 两个矩形面积 - 交集面积
            IoU = area / (parea + garea - area)
            if IoU >= 0.5:
                rightList.append(i)
            # print("IoU是：{}".format(IoU))
        # print(rightList)
        t1 = []
        seriesRightList = []
        for x in rightList:
            t1.append(x)
            if x + 1 not in rightList:
                seriesRightList.append([t1[0], t1[-1]])
                print(t1[0], t1[-1])
                t1 = []
        # 取出连续跟对列表
        print(seriesRightList)
        seriesRightNum = 0
        for i in seriesRightList:
            seriesRightNum += (i[1] - i[0] + 1)
        print("连续跟对的帧数是：{}".format(seriesRightNum))
    else:
        # LK
        # rightList = []
        # for i in range(len(imgRecList)):
        #     p = templateRecList[i]
        #     g = imgRecList[i]
        #     print(intersection(p, g))
        #     if intersection(p, g) >= 0.5:
        #         rightList.append(i)
        width = int(imgRecList[0][2])
        height = int(imgRecList[0][3])
        rightList = []
        for i in range(len(imgRecList)):
            px1, py1, px2, py2 = int(templateRecList[i][1][0]), int(templateRecList[i][1][1]), int(
                templateRecList[i][3][0]), int(templateRecList[i][3][1])
            # print("预测框P的坐标是：({}, {}, {}, {})".format(px1, py1, px2, py2))

            gx1, gy1, gx2, gy2 = int(imgRecList[i][0]), int(imgRecList[i][1]), int(imgRecList[i][0]) + int(
                imgRecList[i][2]), int(imgRecList[i][1]) + int(imgRecList[i][3])
            # print("原标记框G的坐标是：({}, {}, {}, {})".format(gx1, gy1, gx2, gy2))

            parea = (px2 - px1) * (py2 - py1)  # 计算P的面积
            garea = (gx2 - gx1) * (gy2 - gy1)  # 计算G的面积
            # print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

            # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
            x1 = max(px1, gx1)  # 得到左上顶点的横坐标
            y1 = max(py1, gy1)  # 得到左上顶点的纵坐标
            x2 = min(px2, gx2)  # 得到右下顶点的横坐标
            y2 = min(py2, gy2)  # 得到右下顶点的纵坐标

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                rightList.append(i)
                continue

            area = w * h  # G∩P的面积
            # print("G∩P的面积是：{}".format(area))

            # 并集的面积 = 两个矩形面积 - 交集面积
            IoU = area / (parea + garea - area)
            if IoU >= 0.5:
                rightList.append(i)

        t1 = []
        seriesRightList = []
        for x in rightList:
            t1.append(x)
            if x + 1 not in rightList:
                seriesRightList.append([t1[0], t1[-1]])
                print(t1[0], t1[-1])
                t1 = []
        # 取出连续跟对列表
        print(seriesRightList)
        seriesRightNum = 0
        for i in seriesRightList:
            seriesRightNum += (i[1] - i[0] + 1)
        print("连续跟对的帧数是：{}".format(seriesRightNum))
        # pass

    pass
