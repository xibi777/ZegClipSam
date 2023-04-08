import os.path as osp
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.pipelines import Compose, LoadAnnotations
from PIL import Image
import cv2

@DATASETS.register_module()
class BinaryCOCO2014Dataset(CustomDataset):
    """COCO2014 dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = (
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
        'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    PALETTE = [[0,0,0], [64, 128, 0], [128, 160, 192], [128, 192, 192], [192, 160, 128], [128, 32, 128], [0, 0, 32], 
    [192, 128, 96], [128, 224, 128], [192, 128, 160], [192, 0, 224], [0, 128, 96], [192, 64, 32], [0, 0, 224], 
    [128, 128, 224], [192, 64, 128], [128, 64, 0], [0, 0, 160], [0, 0, 128], [128, 32, 192], [128, 128, 0], 
    [64, 0, 160], [64, 192, 192], [0, 192, 64], [64, 128, 32], [128, 32, 64], [0, 32, 192], [192, 64, 160], 
    [128, 64, 192], [192, 0, 64], [0, 224, 128], [128, 0, 0], [64, 0, 64], [0, 64, 96], [0, 192, 160], 
    [192, 32, 64], [192, 0, 0], [0, 192, 0], [0, 224, 192], [0, 64, 192], [192, 0, 32], [128, 128, 32], 
    [128, 64, 128], [128, 160, 128], [64, 0, 96], [128, 192, 160], [128, 96, 0], [0, 128, 160], [128, 0, 32], 
    [0, 128, 192], [192, 0, 160], [192, 192, 192], [128, 64, 64], [128, 192, 0], [64, 32, 0], [64, 32, 128], 
    [128, 0, 192], [0, 96, 0], [0, 224, 0], [0, 160, 192], [64, 64, 192], [128, 128, 64], [192, 32, 128], 
    [64, 32, 192], [0, 64, 96], [64, 0, 0], [192, 160, 0], [64, 128, 160], [0, 32, 0], [64, 160, 0], [192, 0, 192], 
    [128, 96, 192], [0, 192, 224], [192, 128, 192], [128, 64, 32], [0, 64, 160], [0, 128, 64], [64, 192, 96], 
    [64, 0, 128], [64, 64, 32], [128, 192, 64]]

    def __init__(self, split, **kwargs):
        super(BinaryCOCO2014Dataset, self).__init__(
            img_suffix='.jpg', 
            seg_map_suffix='.png', 
            split=split, 
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    def evaluate(self, results, qry_txt_path, num_novel, split, **kwargs):
        # modify ground truth
        self.num_novel = num_novel
        self.split = split

        if self.split == 0:
            self.novel_class = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77]
        elif self.split == 1:
            self.novel_class = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78]
        elif self.split == 2:
            self.novel_class = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79]
        elif self.split == 3:
            self.novel_class = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]

        # for logger
        print('\n' +  '+++++++++++ Novel pair-wise +++++++++++++')
        binary_results = results
        # binary_results = self.get_binary_pred(results, th=0.4)
        targets = self.get_binary_label(qry_txt_path)
        # mIoU = self.iou_mean(binary_results, binary_labels)
        mIoU = 0
        total = len(results)
        for i in range(len(results)):
            intersection, union, new_target = self.intersectionAndUnion(binary_results[i], targets[i], 1)
            mIoU += intersection/union

        print('Total mIoU of 1000 pair-wise novel classes:', mIoU/total)
        return mIoU/total

    def get_binary_pred(self, results, th=0.5):
        binary_results = []
        for i in range(len(results)):
            result = results[i]
            # print('i:', i)
            # print(result.min())
            # print(result.max())
            binary_result = np.zeros_like(result)
            binary_result[result>th] = 1
            binary_results.append(binary_result)
        return binary_results

    def get_binary_label(self, qry_path):
        f = open(qry_path, 'r')
        binary_labels = []
        n = 0
        total_num_img = 1000
        cls_num_img = int(total_num_img/self.num_novel) # voc:200, coco:50

        for filename in f:
            n_cls = self.novel_class[int(n/cls_num_img)] #200!
            filename = filename.strip('\n')
            label_path = '/media/data/ziqin/data_fss/coco2014/Annotations/val_contain_crowd/' + str(filename) + '.png'
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # label[label==0] = 255 ## ignore the ground truth label
            # label[label!=255] -= 1
            binary_label = np.zeros_like(label)
            binary_label[label == n_cls] = 1
            binary_label[label==255] = 255
            binary_labels.append(binary_label)
            n+=1
        return binary_labels

    def intersectionAndUnion(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.ndim in [1, 2, 3])
        assert output.shape == target.shape
        output = output.reshape(output.size).copy()
        target = target.reshape(target.size)
        output[np.where(target == ignore_index)[0]] = 255
        intersection = output[np.where(output == target)[0]]
        area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
        area_output, _ = np.histogram(output, bins=np.arange(K+1))
        area_target, _ = np.histogram(target, bins=np.arange(K+1))
        area_union = area_output + area_target - area_intersection
        return area_intersection, area_union, area_target