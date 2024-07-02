from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths

import os
from progress.bar import Bar

# from external.nms import soft_nms
from opts import opts
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from utils.Density_Adaptive_Algorithm import post_processs


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Detector = detector_factory[opt.task]
    print(opt)

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        img_info = dataset.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(dataset.img_dir, img_info['file_name'])

        ret = detector.run(img_path)
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()

if __name__ == '__main__':
    opt = opts().parse()
    opt.load_model = 'experiments/Dubai/model_100.pth'
    test(opt)
    post_processs(img_path='D:/MOT/DXB/1460-1400/img/')