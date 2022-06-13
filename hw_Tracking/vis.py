import os

import cv2
import numpy as np
import T_SSD
import eval

imgPath = './BlurCar2/img/'
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


def start(imgRecList, imgFiles, templateImg):
    templateImgRec = imgRecList[0]
    croppedTemplateImg = templateImg[int(templateImgRec[1]):int(templateImgRec[1]) + int(templateImgRec[3]),
                         int(templateImgRec[0]):int(templateImgRec[0]) + int(templateImgRec[2])]  # img[y0:y1, x0:x1]模板图
    width = int(templateImgRec[2])
    height = int(templateImgRec[3])
    cv2.rectangle(templateImg, (int(templateImgRec[0]), int(templateImgRec[1])),
                  (int(templateImgRec[0]) + int(templateImgRec[2]),
                   int(templateImgRec[1]) + int(templateImgRec[3])), (0, 0, 255), 2)  # 画模板框，对角点，颜色，线宽
    for i in range(0, len(imgFiles)):
        targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
        targetImgRec = imgRecList[i]  # 目标图的框
        # cv2.rectangle(targetImg, (int(targetImgRec[0]), int(targetImgRec[1])),
        #               (int(targetImgRec[0]) + int(targetImgRec[2]),
        #                int(targetImgRec[1]) + int(targetImgRec[3])), (0, 255, 255), 2)  # 画真实框，黄色，对角点，颜色，线宽
        # cv2.imshow('templateImg', templateImgRecIn)
        # cv2.imshow('targetImg', targetImgRecIn)
        # cv2.waitKey(0)
        match_x, match_y = T_SSD.img_ssd(croppedTemplateImg, targetImg)
        # matchImg = cv2.rectangle(targetImg, (match_x, match_y), (match_x + width, match_y + height), (0, 0, 255), 2)
        # cv2.imshow('matchImg', matchImg)
        print('match_x:', match_x, 'match_y:', match_y, "第", i + 1, "张图片")
        with open('./BlurCar2/result.txt', 'a') as f:
            f.write(str(match_x) + '\t' + str(match_y) + '\n')
        return width, height
        # cv2.waitKey(0)
        # for imgFile in imgFiles:
        #     img = cv2.imread(imgPath + imgFile)
        #     cv2.imshow('img', img)

    pass


def visible(imgFiles, imgRecList, width=0, height=0):
    resultPath = './BlurCar2/result.txt'
    templateList = []
    width = int(imgRecList[0][2])
    height = int(imgRecList[0][3])
    with open(resultPath, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            templateList.append(line)
            print(line)
    for i in range(0, 5):
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
    # width, height = start(imgRecList, imgFiles, templateImg)
    templateList = visible(imgFiles, imgRecList)
    eval.evalMatch(templateList, imgRecList)
