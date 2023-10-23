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
# from mmseg.datasets.pipelines import Compose, LoadAnnotations
from PIL import Image
import cv2
from mmseg.datasets.builder import PIPELINES

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
        
        split_type = int(split.split('_').pop(-1).split('.').pop(0))
        self.train_list='/media/data/ziqin/data_fss/coco2014/ImageSets/BinaryFewShotSegmentation/class_perimage.npy'
        if split_type == 0:
            self.base_class = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 
                28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 48, 50, 51, 52, 54, 
                55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 72, 74, 75, 76, 78, 79, 80]
        elif split_type == 1:
            self.base_class = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27,
                28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 45, 47, 48, 49, 51, 52, 
                53, 55, 56, 57, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 79, 80]
        elif split_type == 2:   
            self.base_class = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 
                29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 
                56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 80]
        elif split_type == 3:   
            self.base_class = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 
                29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 
                55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79]
            
        assert osp.exists(self.img_dir) and self.split is not None
        
        self.class_perimage = np.load(self.train_list, allow_pickle=True)
        
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # self.pre_pipeline(results)
        
        # choose_class = self.base_class[1:][idx % (len(self.base_class) - 1)] #exclude 0 class: bg
        choose_class = self.base_class[1:][np.random.choice((len(self.base_class) - 1), 1, replace=False)[0]]
        assert choose_class != 0
        shot = 10 ## which is better?
        choose_idx = np.random.choice(len(self.class_perimage[choose_class - 1]), shot+1, replace=False) # the last one is query image
        results = self.get_img_ann_info(choose_class, choose_idx)
        return results
    
    def get_img_ann_info(self, choose_class, choose_idx):
        # forquery
        img_info = dict(filename=self.class_perimage[choose_class - 1][choose_idx[0]] + '.jpg', ann=dict(seg_map=self.class_perimage[choose_class - 1][choose_idx[0]] + '.png'))
        ann_info = dict(seg_map=self.class_perimage[choose_class - 1][choose_idx[0]] + '.png')
        results = dict(img_info=img_info, ann_info=ann_info)
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map
        results = self.pipeline(results)
        ## the ground truth should only remain the target label and should be as binary mask seg
        # binary_gt = results['gt_semantic_seg'].data.clone()
        # binary_gt[results['gt_semantic_seg'].data!=choose_class] = 255
        # binary_gt[results['gt_semantic_seg'].data==0] = 0
        # results['gt_semantic_seg'].data[:] = binary_gt[:]
        results['gt_semantic_seg'].data[results['gt_semantic_seg'].data!=choose_class] = 0
        results['gt_semantic_seg'].data[results['gt_semantic_seg'].data==choose_class] = 1
        
        results_support = []
        for i in range(choose_idx.shape[0]-1):
            img_info = dict(filename=self.class_perimage[choose_class - 1][choose_idx[1+i]] + '.jpg', ann=dict(seg_map=self.class_perimage[choose_class - 1][choose_idx[1+i]] + '.png'))
            ann_info = dict(seg_map=self.class_perimage[choose_class - 1][choose_idx[1+i]] + '.png')
            
            support_results = dict(img_info=img_info, ann_info=ann_info)
            
            support_results['seg_fields'] = []
            support_results['img_prefix'] = self.img_dir
            support_results['seg_prefix'] = self.ann_dir
            if self.custom_classes:
                support_results['label_map'] = self.label_map
            support_results = self.pipeline(support_results)
            ## the ground truth should only remain the target label and should be as binary mask seg
            # binary_gt = support_results['gt_semantic_seg'].data.clone()
            # binary_gt[support_results['gt_semantic_seg'].data!=choose_class] = 255
            # binary_gt[support_results['gt_semantic_seg'].data==0] = 0
            # support_results['gt_semantic_seg'].data[:] = binary_gt[:]
            support_results['gt_semantic_seg'].data[support_results['gt_semantic_seg'].data!=choose_class] = 0
            results_support.append(support_results)
        results['support_info'] = results_support
            
        return results
            
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
            intersection, union, new_target = self.intersectionAndUnion(binary_results[i], targets[i], 2)
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