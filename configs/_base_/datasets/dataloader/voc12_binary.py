import os.path as osp
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

import torch
import cv2

@DATASETS.register_module()
class BinaryPascalVOCDataset20(CustomDataset):
    """Pascal VOC dataset.
    Args:
        split (str): Split txt file for Pascal VOC and exclude "background" class.
    """

    CLASSES = ('bakcground', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    PALETTE = [[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, **kwargs):
        super(BinaryPascalVOCDataset20, self).__init__(
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
            # intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), targets[i].cpu().numpy(), new_target.cpu().numpy()
            # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            # subcls = subcls[0].cpu().numpy()[0]
            # class_intersection_meter[(subcls-1)%split_gap] += intersection[1]
            # class_union_meter[(subcls-1)%split_gap] += union[1] 

            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            mIoU += intersection/union

        print('Total mIoU of 1000 pair-wise novel classes:', mIoU/total)
        return mIoU/total

    def iou_mean(self, preds, targets, n_classes = 1):
        #n_classes ：the number of classes in your dataset,not including background
        # for mask and ground-truth label, not probability map
        ious = []
        iousSum = 0
        total_num = len(preds)

        # Ignore IoU for background class ("0")
        # from PIL import Image
        # import matplotlib.pyplot as plt
        # plt.imshow(pred.squeeze()) / plt.imshow(target.squeeze())
        # plt.savefig('1.png') / plt.savefig('2.png')

        for i in range(total_num):
            pred = torch.from_numpy(preds[i])
            pred = pred.view(-1)
            target = torch.from_numpy(targets[i])
            target = target.view(-1)
            assert pred.shape == target.shape

            for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
                pred_inds = pred == cls
                target_inds = target == cls
                intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
                union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
                if union == 0:
                    ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
                else:
                    ious.append(float(intersection) / float(max(union, 1)))
                    iousSum += float(intersection) / float(max(union, 1))
                print('iou:', float(intersection) / float(max(union, 1)))
            miou = iousSum/total_num
        return miou/n_classes

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


    # def get_binary_label(self, qry_path):
    #     f = open(qry_path, 'r')
    #     np_binary_labels = []
    #     n = 0
    #     for filename in f:
    #         n_cls = int(n/2) #200!
    #         filename = filename.strip('\n')
    #         label_path = '/media/data/ziqin/data_fss/VOC2012/Annotations/' + str(filename) + '.png'
    #         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    #         label[label==0] = 255 ## ignore the ground truth label
    #         label[label!=255] -= 1
    #         binary_label = np.zeros_like(label)
    #         binary_label[label == n_cls] = 1
    #         np_binary_labels.append(binary_label)
    #         n+=1
    #     return np_binary_labels

    def get_binary_label(self, qry_path):
        f = open(qry_path, 'r')
        binary_labels = []
        n = 0
        total_num_img = 1000
        cls_num_img = int(total_num_img/self.num_novel) # voc:200, coco:50

        for filename in f:
            n_cls = self.num_novel*self.split + int(n/cls_num_img) + 1 #200! ## +1 for including bg cls
            filename = filename.strip('\n')
            label_path = '/media/data/ziqin/data_fss/VOC2012/Annotations/' + str(filename) + '.png'
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # label[label==0] = 255 ## ignore the ground truth label
            # label[label!=255] -= 1
            binary_label = np.zeros_like(label)
            binary_label[label == n_cls] = 1
            
            ## different settings
            # If set the other class into bg and remain ignore as 255 as GFS-Seg setting
            binary_label[label==255] = 255 
            # If set the other class into ingore 255 and remain bg as 0
            # binary_label[label!=n_cls] = 255
            # binary_label[label==0] = 0
            # If set the other pixels all into bg: pass
            
            binary_labels.append(binary_label)
            n+=1
        return binary_labels


    def intersectionAndUnionGPU(self, output, target, K, ignore_index=255):
        # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
        assert (output.dim() in [1, 2, 3])
        #print(output.shape)
        #print(target.shape)
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == ignore_index] = ignore_index
        intersection = output[output == target]
        # https://github.com/pytorch/pytorch/issues/1382
        area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
        area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
        area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
        area_union = area_output + area_target - area_intersection
        return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


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