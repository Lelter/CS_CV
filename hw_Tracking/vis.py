import os

import cv2
import numpy as np
import T_SSD
import T_NCC
import eval
import T_LK
imgPath = './MountainBike/img/'
file = './BlurCar2/groundtruth_rect.txt'


def readImgFile():
    imgRecList = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            imgRecList.append(line)
    imgFiles = os.listdir(imgPath)
    templateImg = cv2.imread(imgPath + imgFiles[0])
    return imgRecList, imgFiles, templateImg


def start(imgRecList, imgFiles, templateImg, function='ssd'):
    templateImgRec = imgRecList[0]
    croppedTemplateImg = templateImg[int(templateImgRec[1]):int(templateImgRec[1]) + int(templateImgRec[3]),
                         int(templateImgRec[0]):int(templateImgRec[0]) + int(
                             templateImgRec[2])]  # img[y0:y1, x0:x1]模板图裁剪
    width = int(templateImgRec[2])
    height = int(templateImgRec[3])
    # cv2.rectangle(templateImg, (int(templateImgRec[0]), int(templateImgRec[1])),
    #               (int(templateImgRec[0]) + int(templateImgRec[2]),
    #                int(templateImgRec[1]) + int(templateImgRec[3])), (0, 0, 255), 2)  # 画模板框，对角点，颜色，线宽
    if function == 'ssd':
        for i in range(0, len(imgFiles)):
            targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
            match_x, match_y = T_SSD.img_ssd(croppedTemplateImg, targetImg)
            print('match_x:', match_x, 'match_y:', match_y, "第", i + 1, "张图片")
            with open('./BlurCar2/result_ssd.txt', 'a') as f:
                f.write(str(match_x) + '\t' + str(match_y) + '\n')
        return width, height
    elif function == 'ncc':
        for i in range(0, len(imgFiles)):
            targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
            match_x, match_y = T_NCC.img_ncc(croppedTemplateImg, targetImg)
            print('match_x:', match_x, 'match_y:', match_y, "第", i + 1, "张图片")
            with open('./BlurCar2/result_ncc.txt', 'a') as f:
                f.write(str(match_x) + '\t' + str(match_y) + '\n')
        return width, height
    elif function == 'LK':
        T_LK.T_LK2(imgPath,croppedTemplateImg,templateImgRec)
        pass

    pass


def visible(imgFiles, imgRecList, width=0, height=0, function='ssd'):
    if function == 'ssd':
        resultPath = './BlurCar2/result_ssd.txt'
    elif function == 'ncc':
        resultPath = './BlurCar2/result_ncc.txt'
    templateList = []
    width = int(imgRecList[0][2])
    height = int(imgRecList[0][3])
    with open(resultPath, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            templateList.append(line)
            # print(line)
    for i in range(0, len(imgFiles)):
        targetImg = cv2.imread(imgPath + imgFiles[i])
        targetImgRec = imgRecList[i]
        cv2.rectangle(targetImg, (int(targetImgRec[0]), int(targetImgRec[1])),
                      (int(targetImgRec[0]) + int(targetImgRec[2]),
                       int(targetImgRec[1]) + int(targetImgRec[3])), (0, 255, 255), 2)
        cv2.rectangle(targetImg, (int(templateList[i][0]), int(templateList[i][1])),
                      (int(templateList[i][0]) + width,
                       int(templateList[i][1]) + height), (0, 0, 255), 2)
        cv2.imshow('matchImg', targetImg)
        cv2.waitKey(0)
    return templateList
    # targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
    # targetImgRec = imgRecList[i]  # 目标图的框
    # cv2.rectangle(targetImg, (int(targetImgRec[0]), int(targetImgRec[1])),
    #               (int(targetImgRec[0]) + int(targetImgRec[2]),
    #                int(targetImgRec[1]) + int(targetImgRec[3])), (0, 255, 255), 2)  # 画真实框，黄色，对角点，颜色，线宽
    pass


if __name__ == '__main__':
    imgRecList, imgFiles, templateImg = readImgFile()
    width, height = start(imgRecList, imgFiles, templateImg, function='LK')
    # templateList = visible(imgFiles, imgRecList)
    # eval.evalMatch(templateList, imgRecList)
