import os

import cv2
import numpy as np
import T_SSD
import T_NCC
import eval
import T_LK
import msvcrt


def readImgFile():
    if scene == 'BlurCar2' or scene == 'Toy':
        imgRecList = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')
                imgRecList.append(line)
        imgFiles = os.listdir(imgPath)
        templateImg = cv2.imread(imgPath + imgFiles[0])
        return imgRecList, imgFiles, templateImg
    elif scene == 'MountainBike':
        imgRecList = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                imgRecList.append(line)
        imgFiles = os.listdir(imgPath)
        templateImg = cv2.imread(imgPath + imgFiles[0])
        return imgRecList, imgFiles, templateImg


def start(imgRecList, imgFiles, templateImg, function='ssd'):
    global templateRec_LK
    templateImgRec = imgRecList[0]
    x0 = int(templateImgRec[1])
    x1 = int(templateImgRec[1]) + int(templateImgRec[3])
    y0 = int(templateImgRec[0])
    y1 = int(templateImgRec[0]) + int(templateImgRec[2])
    croppedTemplateImg = templateImg[x0:x1, y0:y1]  # img[y0:y1, x0:x1]模板图裁剪
    width = int(templateImgRec[2])
    height = int(templateImgRec[3])
    # cv2.rectangle(templateImg, (int(templateImgRec[0]), int(templateImgRec[1])),
    #               (int(templateImgRec[0]) + int(templateImgRec[2]),
    #                int(templateImgRec[1]) + int(templateImgRec[3])), (0, 0, 255), 2)  # 画模板框，对角点，颜色，线宽
    if function == 'ssd':
        with open('./{}/result_ssd_{}.txt'.format(scene, scene), 'a') as f:
            for i in range(0, len(imgFiles)):
                targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
                match_x, match_y = T_SSD.img_ssd(croppedTemplateImg, targetImg)
                croppedTemplateImg = targetImg[match_y:match_y + height, match_x:match_x + width]
                print('match_x:', match_x, 'match_y:', match_y, "第", i + 1, "张图片")
                f.write(str(match_x) + '\t' + str(match_y) + '\n')
        return width, height
    elif function == 'ncc':
        with open('./{}/result_ncc_{}.txt'.format(scene, scene), 'w') as f:
            for i in range(0, len(imgFiles)):
                targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
                match_x, match_y = T_NCC.img_ncc(croppedTemplateImg, targetImg)
                croppedTemplateImg = targetImg[match_y:match_y + height, match_x:match_x + width]
                print('match_x:', match_x, 'match_y:', match_y, "第", i + 1, "张图片")
                f.write(str(match_x) + '\t' + str(match_y) + '\n')
        return width, height
    elif function == 'LK':
        templateRec_LK = T_LK.T_LK(imgPath, croppedTemplateImg, templateImgRec)
        return width, height
        pass

    pass


def visible(imgFiles, imgRecList, width=0, height=0, function='ssd'):
    if function == 'ssd':
        resultPath = './{}/result_ssd_{}.txt'.format(scene, scene)
    elif function == 'ncc':
        resultPath = './{}/result_ncc_{}.txt'.format(scene, scene)
    if function == 'ncc' or function == 'ssd':
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
            cv2.waitKey(100)
        return templateList
    else:
        templateList = []
        width = int(imgRecList[0][2])
        height = int(imgRecList[0][3])
        for i in range(len(imgRecList)):
            targetImg = cv2.imread(imgPath + imgFiles[i])
            targetImgRec = imgRecList[i]
            cv2.rectangle(targetImg, (int(targetImgRec[0]), int(targetImgRec[1])),
                          (int(targetImgRec[0]) + int(targetImgRec[2]),
                           int(targetImgRec[1]) + int(targetImgRec[3])), (0, 255, 255), 2)
            print('match_x:', templateRec_LK[i][0], 'match_y:', templateRec_LK[i][1])
            x_mean = 0
            y_mean = 0
            for j in range(len(templateRec_LK[i])):
                x_mean += templateRec_LK[i][j][0]
                y_mean += templateRec_LK[i][j][1]
            x_mean /= len(templateRec_LK[i])
            y_mean /= len(templateRec_LK[i])
            x_mean = int(x_mean)
            y_mean = int(y_mean)
            print('x_mean:', x_mean, 'y_mean:', y_mean)
            temppoint = [[x_mean - width / 2, y_mean + height / 2], [x_mean - width / 2, y_mean - height / 2],
                         [x_mean + width / 2, y_mean - height / 2], [x_mean + width / 2, y_mean + height / 2]]
            templateList.append(temppoint)
            cv2.circle(targetImg, (int(x_mean), int(y_mean)), 5, (0, 0, 255), 2)
            cv2.rectangle(targetImg, (int(x_mean - width / 2), int(y_mean - height / 2)),
                          (int(x_mean + width / 2),
                           int(y_mean + height / 2)), (0, 0, 255), 2)
            # minAreaRect = cv2.minAreaRect(np.array(templateRec_LK[i]))
            # box = cv2.boxPoints(minAreaRect)
            # box = np.int0(box)
            #
            # left_point_x = np.min(box[:, 0])
            # right_point_x = np.max(box[:, 0])
            # top_point_y = np.max(box[:, 1])
            # bottom_point_y = np.min(box[:, 1])
            # left_point_y = sorted(box[:, 1][np.where(box[:, 0] == left_point_x)])[-1]
            # right_point_y = sorted(box[:, 1][np.where(box[:, 0] == right_point_x)])[0]
            # top_point_x = sorted(box[:, 0][np.where(box[:, 1] == top_point_y)])[-1]
            # bottom_point_x = sorted(box[:, 0][np.where(box[:, 1] == bottom_point_y)])[0]
            #
            # vertices = [[left_point_x, left_point_y], [bottom_point_x, bottom_point_y],
            #             [right_point_x, right_point_y], [top_point_x, top_point_y]]
            # templateList.append(vertices)
            # cv2.drawContours(targetImg, [box], 0, (0, 0, 255), 2)
            #
            # for (x, y) in templateRec_LK[i]:
            #     cv2.circle(targetImg, (x, y), 2, (0, 0, 255), 1)
            # # cv2.circle(targetImg, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.imshow('matchImg', targetImg)
            cv2.waitKey(1)
        return templateList
        pass
    # targetImg = cv2.imread(imgPath + imgFiles[i])  # 目标图
    # targetImgRec = imgRecList[i]  # 目标图的框
    # cv2.rectangle(targetImg, (int(targetImgRec[0]), int(targetImgRec[1])),
    #               (int(targetImgRec[0]) + int(targetImgRec[2]),
    #                int(targetImgRec[1]) + int(targetImgRec[3])), (0, 255, 255), 2)  # 画真实框，黄色，对角点，颜色，线宽


if __name__ == '__main__':
    scene = 'BlurCar2'
    imgPath = './{}/img/'.format(scene)
    file = './{}/groundtruth_rect.txt'.format(scene)
    templateRec_LK = []
    good_list = []
    p1_list = []
    function = 'LK'
    imgRecList, imgFiles, templateImg = readImgFile()
    width, height = start(imgRecList, imgFiles, templateImg, function=function)
    # print(templateRec_LK)
    templateList = visible(imgFiles, imgRecList, function=function)
    eval.evalMatch(templateList, imgRecList, function=function)
