import PIL.Image as Image
import os
import cv2
import numpy as np
import time
import PIL.Image as Image
from PIL import ImageDraw


def post_processs(img_path, thres_all=100, thres_center=50, n_frame=50, seg_weight=0.2, ):
    f = open('output\\result.txt', 'w')
    for img_id in range(1, n_frame + 1):
        center = Image.open('output/hm_center/%03d.png' % img_id)
        all = Image.open('output/hm_all/%03d.png' % img_id)
        seg = Image.open('output/seg/%03d.png' % img_id)

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
        frame = Image.open(os.path.join(img_path, '%06d.jpg' % img_id))
        draw = ImageDraw.ImageDraw(frame)
        for bbox in stats:
            draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), outline=(0, 0, 255), width=1)
        for center in centroids:
            center = [int(i) for i in center]
            draw.ellipse(((center[0] - 1, center[1] - 1), (center[0] + 1, center[1] + 1)), fill=(255, 0, 0),
                         outline=(255, 0, 0), width=1)
        frame.save('output/vis_pre/%06d.png' % img_id)

        # match & save
        frame = cv2.imread(os.path.join(img_path, '%06d.jpg' % img_id))
        for bbox in stats:
            if bbox[4] > 5000 or bbox[4] < 36:
                continue
            candidate = []
            for center in centroids:
                if bbox[0] < center[0] < bbox[0] + bbox[2] and bbox[1] < center[1] < bbox[1] + bbox[3]:
                    candidate.append(center)
            if len(candidate) == 1:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 1)
                f.write(str(img_id) + ','
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
                    f.write(str(img_id) + ','
                            + str(int(center[0] - w)) + ','
                            + str(int(center[1] - h)) + ','
                            + str(int(center[0] + w)) + ','
                            + str(int(center[1] + h)) + '\n')
        cv2.imwrite('output/results/%06d.png' % img_id, frame)

    f.close()


if __name__ == '__main__':
    post_processs(img_path='D:/MOT/DXB/1460-1400/img/')
