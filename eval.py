from pycocotools.coco import COCO
import json
import PIL.Image as Image
from PIL import ImageDraw
import numpy as np
import os


def nms(dets, thresh):
    '''
    :param dets: NMS前的检测结果,输入格式为[right,top,right,bottom,score]
    :param thresh: 小于阈值则抑制
    :return: NMS后的检测结果,输出格式为[right,top,right,bottom,score]
    '''
    # 整理数据,获取信息
    dets = np.array(dets)
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    scores = dets[:, 4]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    # 保存没有被抑制的框的坐标
    remian = []
    # 根据置信度分数从高到底排序
    index = scores.argsort()[::-1]
    while index.size > 0:
        # 选择当前没有被抑制的置信度最高的框的坐标,存入remain
        i = index[0]
        remian.append(i)
        # 计算其余框与该框的IoUs
        X1 = np.maximum(x1[i], x1[index[1:]])
        Y1 = np.maximum(y1[i], y1[index[1:]])
        X2 = np.minimum(x2[i], x2[index[1:]])
        Y2 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, X2 - X1 + 1)
        h = np.maximum(0, Y2 - Y1 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 保留IoU小于阈值的框,也就是抑制IoU大于阈值的框
        idx = np.where(ious <= thresh)[0]
        # 下一次迭代
        index = index[idx + 1]
    return remian


def bbox_iou_single(box1, box2):
    '''
    box:[top, left, bottom, right]
    '''
    inter_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if inter_h < 0 or inter_w < 0 else inter_h * inter_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return inter / (union + 1e-10)


def eval():
    iou_thres = 0.3

    tp, fn, fp = 0, 0, 0

    load_path = 'H:/Liang/Datas/MOT'
    # img_path = 'H:/Liang/Datas/MOT/SkySat/001/img'
    img_path = 'H:/Liang/Datas/MOT/DXB/1460-1400/img'
    # img_path = 'H:/Liang/Datas/MOT/SD/9590-2960/img'
    img_list = os.listdir(img_path)
    img_list.remove('000000.jpg')
    img_list.remove('000051.jpg')

    for frame_id in range(15, 20):
        img = Image.open(img_path + '/%06d.jpg' % frame_id)
        draw = ImageDraw.ImageDraw(img)

        gt, pred, match_gt, match_pred = [], [], [], []

        # f = open(load_path + '/SD/9590-2960/gt/gt.txt', 'r')
        f = open(load_path + '/DXB/1460-1400/gt/gt.txt', 'r')
        for line in f:
            line = line.split(',')
            line = [int(i) for i in line]
            if line[0] == frame_id:
                gt.append([line[2], line[3], line[4], line[5]])
        f.close()

        f = open('debug\\result.txt', 'r')
        for line in f:
            line = line.split(',')
            line = [int(i) for i in line]
            if line[0] == frame_id:
                pred.append([line[1], line[2], line[3], line[4]])
        f.close()

        for id_gt, car_gt in enumerate(gt):
            for id_pred, car_pred in enumerate(pred):
                if bbox_iou_single(car_pred, car_gt) > iou_thres \
                        and id_pred not in match_pred \
                        and id_gt not in match_gt:
                    match_pred.append(id_pred)
                    match_gt.append(id_gt)
                    # draw.rectangle(car_gt, fill=None, outline='green', width=1)
                    draw.rectangle(car_pred, fill=None, outline='blue', width=1)
                    tp += 1
            if id_gt not in match_gt:
                draw.rectangle(car_gt, fill=None, outline='yellow', width=1)
                fn += 1

        for id_pred, car_pred in enumerate(pred):
            if id_pred not in match_pred:
                draw.rectangle(car_pred, fill=None, outline='red', width=1)
                fp += 1

        img.save('debug/result/%06d.png' % frame_id, quality=95)
        # print('result/%06d.png' % frame_id)

    try:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        print('Prec: %.5f, Rec: %.5f, F1: %.5f' % (prec, rec, f1))
        print('TP: %d, FP: %d ,FN: %d' % (tp, fp, fn))
    except:
        print('Prec: %d, Rec: %d, F1: %d' % (0, 0, 0))


if __name__ == '__main__':
    eval()
