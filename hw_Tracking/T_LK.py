import os

import cv2
import numpy as np

imgPath = './MountainBike/img/'
file = './MountainBike/groundtruth_rect.txt'

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


def T_LK(imgPath, croppedTemplateImg, templateImgRec):
    track_len = 10
    detect_interval = 5
    tracks_list = []
    tracks = []
    frame_idx = 0
    imgRecList = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            imgRecList.append(line)
    imgFiles = os.listdir(imgPath)

    for i in range(0, len(imgFiles)):
        targetImg = cv2.imread(imgPath + imgFiles[i])
        targetImgGray = cv2.cvtColor(targetImg, cv2.COLOR_BGR2GRAY)
        vis = targetImg.copy()
        if len(tracks) > 0:
            img0 = preTargetImgGray
            img1 = targetImgGray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                    **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
            d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系
            good = d  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
            new_tracks = []
            templist = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                templist.append((int(x), int(y)))

            tracks_list.append(templist)
            if i == 1:
                tracks_list.append(templist)
            templist = []
            tracks = new_tracks

            # cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))  # 以上一振角点为初始点，当前帧跟踪到的点为终点划线
            # draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
        if frame_idx == 0:
            mask = np.zeros_like(targetImgGray)
            if i == 1:
                mask[int(templateImgRec[1]):int(templateImgRec[1]) + int(templateImgRec[3]),
                int(templateImgRec[0]):int(templateImgRec[0]) + int(templateImgRec[2])] = 255
            else:
                mask[:] = 255
            # mask[:]=255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (int(x), int(y)), 5, 0, -1)
            x0 = int(templateImgRec[0])
            x1 = int(templateImgRec[0]) + int(templateImgRec[2])
            y0 = int(templateImgRec[1])
            y1 = int(templateImgRec[1]) + int(templateImgRec[3])
            for i in range(x0, x1):
                for j in range(y0, y1):
                    tracks.append([(i, j)])
            # tracks_list.append(tracks)
            # p=np.array()
            # p = cv2.goodFeaturesToTrack(targetImgGray, mask=mask, **feature_params)
            # if p is not None:
            #     for x, y in np.float32(p).reshape(-1, 2):
            #         tracks.append([(x, y)])
        frame_idx += 1
        preTargetImgGray = targetImgGray
        # cv2.imshow('lk_track', vis)
        # cv2.waitKey(0)
    print(tracks_list)
    return tracks_list
