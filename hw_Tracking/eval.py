import cv2
import numpy


def evalMatch(templateRecList, imgRecList, width=0, height=0):
    # 参数：templateRecList：匹配框的坐标，imgRecList：真实框的坐标列表
    # 跟对的帧数列表
    # TargetRec为(左上角x,y,宽，高)
    # templateRec为(左上角x,y)
    width = int(imgRecList[0][2])
    height = int(imgRecList[0][3])
    rightList = []
    for i in range(len(imgRecList)):
        px1, py1, px2, py2 = int(templateRecList[i][0]), int(templateRecList[i][1]), int(templateRecList[i][0]) + width, \
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
        # print("x1:",x1,"y1:",y1,"x2:",x2,"y2:",y2)
        # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
        # w = max(0, (x2 - x1))  # 相交矩形的长，这里用w来表示
        # h = max(0, (y1 - y2))  # 相交矩形的宽，这里用h来表示
        # print("相交矩形的长是：{}，宽是：{}".format(w, h))
        # 这里也可以考虑引入if判断
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

    pass
