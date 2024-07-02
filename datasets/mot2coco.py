import numpy as np
from PIL import Image
import os
import json


def load_img(load_path, save_path, datasets, split):
    save_id = 1
    save_label = []
    for dataset in datasets:
        file_list = os.listdir(os.path.join(load_path, dataset, 'img'))
        data = np.loadtxt(os.path.join(load_path, dataset, 'gt\gt.txt'), delimiter=',', dtype=int).tolist()
        for file in file_list[1:-1]:
            img_id = int(file[:-4])
            label = [i for i in data if i[0] == img_id]
            img_pre = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id - 1)))
            img = Image.open(os.path.join(load_path, dataset, 'img', file))
            img_post = Image.open(os.path.join(load_path, dataset, 'img', '%06d.jpg' % (img_id + 1)))
            road = Image.open(os.path.join(load_path, dataset, 'road', file)).convert('L')
            img_w, img_h = img.size
            for obj in label:
                x, y = obj[2], obj[3]
                w, h = obj[4] - obj[2], obj[5] - obj[3]
                save_label.append([x, y, w, h, save_id])
            img_save = np.zeros((img_h, img_w, 10))
            img_save[:, :, 0:3] = img_pre
            img_save[:, :, 3:6] = img
            img_save[:, :, 6:9] = img_post
            img_save[:, :, 9] = road
            np.save(os.path.join(save_path, split, '%06d.npy' % save_id),
                    np.array(img_save, dtype=int))
            save_id += 1
            print(os.path.join(save_path, split, file))
    print(save_id, len(save_label))
    return save_label, save_id


def convert_json(data, save_path, split, save_name, img_size=(1000, 1000)):
    coco = dict()
    coco['info'] = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []
    coco["licenses"] = []

    info = {"year": 2023,
            "version": '1.0',
            "description": 'coco format',
            "contributor": 'liang',
            "url": 'null',
            "date_created": '2023.3.30'
            }
    coco['info'] = info

    categories = {"id": 1,
                  "name": 'vehicle',
                  "supercategory": 'null',
                  }
    coco['categories'].append(categories)

    for file in os.listdir(os.path.join(save_path, split)):
        image = {"id": int(file.split('.')[0]),
                 "width": img_size[1],
                 "height": img_size[0],
                 "file_name": file,
                 "license": 0,
                 "flickr_url": 'null',
                 "coco_url": 'null',
                 "date_captured": '2022.4.11'
                 }
        coco['images'].append(image)

    for i, item in enumerate(data):
        bbox = [item[0], item[1], item[2], item[3]]
        annotation = {"id": i,
                      "image_id": item[4],
                      "category_id": 1,
                      "segmentation": [],
                      "area": item[2] * item[3],
                      "bbox": bbox,
                      "iscrowd": 0,
                      'ignore': 0
                      }
        coco['annotations'].append(annotation)
    print(len(coco['images']), len(coco['annotations']))

    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(coco, f)

    return coco


if __name__ == "__main__":
    labels, img_num = load_img(load_path=r'D:\MOT\DXB',
                               save_path=r'D:\MOT\DXB',
                               datasets=['1460-1400'],
                               split='val')
    coco_json = convert_json(data=labels,
                             save_path=r'D:\MOT\DXB',
                             split='val',
                             save_name=r'annotations\val.json',
                             img_size=(1000, 1000))
