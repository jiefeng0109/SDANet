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


def eval(coco, outputs, root_path, con_thres, iou_thres, stride=250,
         datasets=['DXB/1460-1400/img'],
         load_path='H:/Liang/Datas/ORSV/car'):
    n_dataset = 0
    tp, fn, fp = 0, 0, 0
    for dataset in datasets:

        img_path = os.path.join(load_path, dataset)
        img_list = os.listdir(img_path)
        img_list.remove('000000.jpg')
        img_list.remove('000051.jpg')
        img = Image.open(img_path + '/' + img_list[0])
        img_w, img_h = img.size
        row, col = img_w // stride, img_h // stride

        tp = fn = fp = 0

        for i in range(len(img_list)):
            gt, pred, match_gt, match_pred = [], [], [], []

            i = i + len(img_list) * n_dataset

            for img_id in range(i * row * col, (i + 1) * row * col):
                n = (img_id - (i * row * col)) // col
                m = (img_id - (i * row * col)) % col

                gt_img = [coco.anns[i]['bbox'] for i in coco.anns if coco.anns[i]['image_id'] == img_id]
                for car in gt_img:
                    x1 = int(car[0]) + n * stride
                    y1 = int(car[1]) + m * stride
                    x2 = int(car[0] + car[2]) + n * stride
                    y2 = int(car[1] + car[3]) + m * stride
                    gt.append([x1, y1, x2, y2, 1])

                pred_img_bbox = [i['bbox'] for i in outputs if i['image_id'] == img_id]
                pred_img_score = [i['score'] for i in outputs if i['image_id'] == img_id]
                for car_id, car in enumerate(pred_img_bbox):
                    if pred_img_score[car_id] > con_thres:
                        x1 = int(car[0]) + n * stride
                        y1 = int(car[1]) + m * stride
                        x2 = int(car[0] + car[2]) + n * stride
                        y2 = int(car[1] + car[3]) + m * stride
                        pred.append([x1, y1, x2, y2, pred_img_score[car_id]])

            if len(pred) == 0:
                fn += len(gt)
                continue

            keep_pred, keep_gt = nms(pred, 0.1), nms(gt, 0.9)
            pred, gt = [pred[i] for i in keep_pred], [gt[i] for i in keep_gt]
            pred, gt = sorted(pred, key=lambda x: x[4], reverse=True), sorted(gt, key=lambda x: x[4], reverse=True)
            pred, gt = [i[0:4] for i in pred], [i[0:4] for i in gt]

            for id_gt, car_gt in enumerate(gt):
                for id_pred, car_pred in enumerate(pred):
                    if bbox_iou_single(car_pred, car_gt) > iou_thres \
                            and id_pred not in match_pred \
                            and id_gt not in match_gt:
                        match_pred.append(id_pred)
                        match_gt.append(id_gt)
                        tp += 1
                if id_gt not in match_gt:
                    fn += 1

            for id_pred, car_pred in enumerate(pred):
                if id_pred not in match_pred:
                    fp += 1

        n_dataset += 1

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    print('Prec: %.5f, Rec: %.5f, F1: %.5f' % (prec, rec, f1))
    return f1


def vis(coco, outputs, root_path, con_thres, iou_thres, stride=250,
        datasets=['DXB/1460-1400/img'],
        load_path='H:/Liang/Datas/ORSV/car'):
    n_dataset = 0
    save_id = 1
    tp, fn, fp = 0, 0, 0
    for dataset in datasets:
        img_path = os.path.join(load_path, dataset)
        img_list = os.listdir(img_path)

        img_list.remove('000000.jpg')
        img_list.remove('000051.jpg')
        img = Image.open(img_path + '/' + img_list[0])
        img_w, img_h = img.size
        row, col = img_w // stride, img_h // stride

        for i in range(len(img_list)):

            img = Image.open(img_path + '/' + img_list[i])
            draw = ImageDraw.ImageDraw(img)

            gt, pred, match_gt, match_pred = [], [], [], []

            i = i + len(img_list) * n_dataset

            for img_id in range(i * row * col, (i + 1) * row * col):
                n = (img_id - (i * row * col)) // col
                m = (img_id - (i * row * col)) % col

                gt_img = [coco.anns[i]['bbox'] for i in coco.anns if coco.anns[i]['image_id'] == img_id]
                for car in gt_img:
                    x1 = int(car[0]) + n * stride
                    y1 = int(car[1]) + m * stride
                    x2 = int(car[0] + car[2]) + n * stride
                    y2 = int(car[1] + car[3]) + m * stride
                    gt.append([x1, y1, x2, y2, 1])

                pred_img_bbox = [i['bbox'] for i in outputs if i['image_id'] == img_id]
                pred_img_score = [i['score'] for i in outputs if i['image_id'] == img_id]
                for car_id, car in enumerate(pred_img_bbox):
                    if pred_img_score[car_id] > con_thres:
                        x1 = int(car[0]) + n * stride
                        y1 = int(car[1]) + m * stride
                        x2 = int(car[0] + car[2]) + n * stride
                        y2 = int(car[1] + car[3]) + m * stride
                        pred.append([x1, y1, x2, y2, pred_img_score[car_id]])

            if len(pred) == 0:
                continue

            keep_pred, keep_gt = nms(pred, 0.1), nms(gt, 0.9)
            pred, gt = [pred[i] for i in keep_pred], [gt[i] for i in keep_gt]
            pred, gt = sorted(pred, key=lambda x: x[4], reverse=True), sorted(gt, key=lambda x: x[4], reverse=True)
            pred, gt = [i[0:4] for i in pred], [i[0:4] for i in gt]

            for id_gt, car_gt in enumerate(gt):
                for id_pred, car_pred in enumerate(pred):
                    if bbox_iou_single(car_pred, car_gt) > iou_thres \
                            and id_pred not in match_pred \
                            and id_gt not in match_gt:
                        match_pred.append(id_pred)
                        match_gt.append(id_gt)
                        draw.rectangle(car_gt, fill=None, outline='green', width=1)
                        draw.rectangle(car_pred, fill=None, outline='blue', width=1)
                        tp += 1
                if id_gt not in match_gt:
                    draw.rectangle(car_gt, fill=None, outline='yellow', width=1)
                    fn += 1

            for id_pred, car_pred in enumerate(pred):
                if id_pred not in match_pred:
                    draw.rectangle(car_pred, fill=None, outline='red', width=1)
                    fp += 1

            img.save(root_path + '/result/%06d.png' % save_id, quality=95)
            print(root_path + '/result/%06d.png' % save_id)
            save_id += 1

        n_dataset += 1
        # break

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    print('Prec: %.5f, Rec: %.5f, F1: %.5f' % (prec, rec, f1))

    return


def mosaic():
    root_path = 'experiments/rnn-seg-DXB-master'
    coco = COCO(annotation_file='H:/Liang/Datas/COCO/Car/DXB_seg/annotations/instances_val.json')

    with open(root_path + '/results.json', 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        if len(strF) > 0:
            outputs = json.loads(strF)

    # con = 0.10
    # max_f1 = -1
    # max_con = 0
    # for iter in range(8):
    #     print('con: %.2f' % con, '  ', end=' ')
    #     f1 = eval(con_thres=con, iou_thres=0.3, root_path=root_path, coco=coco, outputs=outputs)
    #     if f1 < 0.1:
    #         break
    #     if max_f1 - f1 > 0.01:
    #         break
    #     if f1 > max_f1:
    #         max_f1 = f1
    #         max_con = con
    #     con += 0.05
    # print('max_con: %.2f, max_f1: %.5f' % (max_con, max_f1))

    vis(con_thres=0.15, iou_thres=0.3, root_path=root_path, coco=coco, outputs=outputs)


if __name__ == "__main__":
    mosaic()
