import PIL.Image as Image
import os
import cv2
import numpy as np
from eval import eval
import time
import PIL.Image as Image
from PIL import ImageDraw


def post_processs(thres_all=100, thres_center=50, n_frame=50, seg_weight=0.2):
    f = open('debug\\result.txt', 'w')

    # for i in range(0, n_frame):
    #     w, h = 500, 500
    #     k = 0
    #     road = np.zeros((w, h))
    #     for j in range(max(0, i - 5), min(60, i + 5)):
    #         k += 1
    #         all = cv2.imread('debug/hm_mosaic_all/%03d.png' % j, cv2.IMREAD_GRAYSCALE)
    #         road += np.array(all, 'float32')
    #     if k == 0:
    #         print(i)
    #     road /= k
    #     road[road < 30] = 0
    #     cv2.imwrite('debug/road/%03d.png' % i, road)

    for frame_id in range(0, n_frame):

        # mosaic
        w, h = 1500, 700
        id_stride = 8
        w_stride, h_stride = 4, 2
        p_stride = 128

        img_id = frame_id * id_stride
        center = Image.new('L', (w, h))
        for y in range(0, w_stride):
            for x in range(0, h_stride):
                image = Image.open('debug/hm_center/%03d.png' % img_id)
                center.paste(image, (p_stride * y, p_stride * x))
                img_id += 1
        center.save('debug/hm_center_mosaic/%03d.png' % frame_id)

        img_id = frame_id * id_stride
        all = Image.new('L', (w, h))
        for y in range(0, w_stride):
            for x in range(0, h_stride):
                image = Image.open('debug/hm_all/%03d.png' % img_id)
                all.paste(image, (p_stride * y, p_stride * x))
                img_id += 1
        all.save('debug/hm_all_mosaic/%03d.png' % frame_id)

        img_id = frame_id * id_stride
        seg = Image.new('L', (w, h))
        for y in range(0, w_stride):
            for x in range(0, h_stride):
                image = Image.open('debug/seg/%03d.png' % img_id)
                seg.paste(image, (p_stride * y, p_stride * x))
                img_id += 1
        seg.save('debug/seg_mosaic/%03d.png' % frame_id)

        # seg = cv2.imread('debug/road/%03d.png' % frame_id, cv2.IMREAD_GRAYSCALE)

        # center = cv2.imread('debug/hm_center_mosaic/%03d.png' % frame_id, cv2.IMREAD_GRAYSCALE)
        # all = cv2.imread('debug/hm_all_mosaic/%03d.png' % frame_id, cv2.IMREAD_GRAYSCALE)
        # seg = cv2.imread('debug/seg_mosaic/%03d.png' % frame_id, cv2.IMREAD_GRAYSCALE)

        # upsampling
        center = np.array(center)
        all = np.array(all)
        seg = np.array(seg)
        center = cv2.resize(center, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        all = cv2.resize(all, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(seg, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        seg[seg < 80] = 0
        all = all * (1 - seg_weight) + seg * seg_weight
        all[all < thres_all] = 0
        all[all != 0] = 255
        all = np.array(all, 'int8')

        center[center < thres_center] = 0
        NpKernel = np.uint8(np.ones((3, 3)))
        center = cv2.erode(center, NpKernel)
        center[center != 0] = 255
        center = np.array(center, 'int8')

        # bbox
        _, _, stats, _ = cv2.connectedComponentsWithStats(all)

        # center
        _, _, _, centroids = cv2.connectedComponentsWithStats(center)

        # # vis_pre
        # frame = Image.open('H:/Liang/Datas/MOT/SD/9590-2960/img/%06d.jpg' % (frame_id + 1))
        frame = Image.open('H:/Liang/Datas/MOT/DXB/1460-1400/img/%06d.jpg' % (frame_id + 1))
        draw = ImageDraw.ImageDraw(frame)
        for bbox in stats:
            draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), outline=(0, 0, 255), width=1)
        for center in centroids:
            center = [int(i) for i in center]
            draw.ellipse(((center[0] - 1, center[1] - 1), (center[0] + 1, center[1] + 1)), fill=(255, 0, 0),
                         outline=(255, 0, 0), width=1)
        # cv2.imwrite('debug/vis_pre/%06d.png' % (frame_id + 1), frame)
        frame.save('debug/vis_pre/%06d.png' % (frame_id + 1))
        print('vis_pre/%06d.png' % (frame_id + 1))

        # match & save
        # frame = cv2.imread('H:/Liang/Datas/MOT/SD/9590-2960/img/%06d.jpg' % (frame_id + 1))
        frame = cv2.imread('H:/Liang/Datas/MOT/DXB/1460-1400/img/%06d.jpg' % (frame_id + 1))
        for bbox in stats:
            if bbox[4] > 5000 or bbox[4] < 36:
                continue
            candidate = []
            for center in centroids:
                if bbox[0] < center[0] < bbox[0] + bbox[2] and bbox[1] < center[1] < bbox[1] + bbox[3]:
                    candidate.append(center)
            if len(candidate) == 1:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)
                f.write(str(frame_id + 1) + ','
                        + str(bbox[0]) + ','
                        + str(bbox[1]) + ','
                        + str(bbox[0] + bbox[2]) + ','
                        + str(bbox[1] + bbox[3]) + '\n')
            elif len(candidate) > 1:
                for center in candidate:
                    right = center[0] - bbox[0]
                    left = bbox[0] + bbox[2] - center[0]
                    top = center[1] - bbox[1]
                    bottom = bbox[1] + bbox[3] - center[1]
                    w = min(right, left, 5)
                    h = min(top, bottom, 5)
                    w = max(w, 3)
                    h = max(h, 3)
                    frame = cv2.rectangle(frame,
                                          (int(center[0] - w), int(center[1] - h)),
                                          (int(center[0] + w), int(center[1] + h)),
                                          (255, 0, 0), 1)
                    f.write(str(frame_id + 1) + ','
                            + str(int(center[0] - w)) + ','
                            + str(int(center[1] - h)) + ','
                            + str(int(center[0] + w)) + ','
                            + str(int(center[1] + h)) + '\n')

    f.close()


if __name__ == '__main__':

    post_processs(thres_all=100, thres_center=60, seg_weight=0.2)
    eval()