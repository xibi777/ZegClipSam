from importlib.resources import path
from unittest.mock import patch
from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
# from mmseg.models.segmentors.base import BaseSegmentor
# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from .few_base import FewBaseSegmentor
from .few_encoder_decoder import FewEncoderDecoder
from mmcv.runner import BaseModule, auto_fp16

from .untils import tokenize
from utils import transform

import numpy as np
import tqdm
import cv2
import itertools

import os
import matplotlib.pyplot as plt

@SEGMENTORS.register_module()
class FewSegViT(FewEncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 pretrained,
                 context_length,
                 base_class,
                 novel_class,
                 both_class,
                 split,
                 shot,
                 supp_dir=None,
                 supp_path=None, # only for evluation on base+novel classes
                 tau=0.07,
                 ft_backbone=False,
                 exclude_key=None,
                #  init_cfg=None,
                 **args):
        super(FewSegViT, self).__init__(**args)
        
        self.pretrained = pretrained

        self.tau = tau

        self.base_class = np.asarray(base_class)
        self.novel_class = np.asarray(novel_class)
        self.both_class = np.asarray(both_class)

        self.split = split
        self.shot = shot
        self.supp_dir = supp_dir
        self.supp_path = supp_path # only for evaluation

        if len(self.base_class) != len(self.both_class): # few-shot setting
            self._visiable_mask(self.base_class, self.novel_class, self.both_class)

        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]
        self.val_supp_transform = transform.Compose([
            transform.Resize(size=max(512, 512)),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        self.val_supp_aug_transform = transform.Compose([
            transform.RandScale(scale=[0.75, 1.5]),
            # transform.Resize(size=max(784, 784)),
            # transform.Crop(size=(512, 512), crop_type='rand'),
            transform.Resize(size=max(512, 512)),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

        if self.training:
            if ft_backbone is False:
                self._freeze_stages(self.backbone, exclude_key=exclude_key)
        else:
            self.backbone.eval()

    def _freeze_stages(self, model, exclude_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if exclude_key:
                if isinstance(exclude_key, str):
                    if not exclude_key in n:
                        m.requires_grad = False
                elif isinstance(exclude_key, list):
                    count = 0
                    for i in range(len(exclude_key)):
                        i_layer = str(exclude_key[i])
                        if i_layer in n:
                            count += 1
                    if count == 0:
                        m.requires_grad = False
                    elif count > 0:
                        print('Finetune layer in backbone:', n)
                else:
                    assert AttributeError("Dont support the type of exclude_key!")
            else:
                m.requires_grad = False

    def _visiable_mask(self, seen_classes, novel_classes, both_classes):
        seen_map = np.array([255]*256)
        for i, n in enumerate(list(seen_classes)):
            seen_map[n] = i
        self.visibility_seen_mask = seen_map.copy()
        print('Making visible mask for zero-shot setting:', self.visibility_seen_mask) 
    
    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _decode_head_forward_train(self, feat, img_metas, gt_semantic_seg, qs_epoch=None):
        """Run forward function and calculate loss for decode head in
        training."""

        losses = dict()
        
        loss_decode = self.decode_head.forward_train(feat, 
                                                    img_metas,
                                                    gt_semantic_seg,
                                                    qs_epoch,
                                                    self.train_cfg,)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
        
    def _decode_head_forward_test(self, x, img_metas, novel_queries=None):
        """Run forward function and calculate loss for decode head in
        inference."""
        if novel_queries is not None:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg, novel_queries)
        else:
            seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            ## register novel prototypies:
            if len(self.base_class) != len(self.both_class): #generalized few-shot setting
                if not hasattr(self, 'novel_queries'):
                    print('\n' + '------------Registering the prototypes of novel classes-----------')
                    self.novel_queries = self.extract_novel_proto(self.supp_dir, self.supp_path, way=len(self.novel_class), shot=self.shot)
                    # print('------------Registering the aug prototypes of novel classes-----------')
                    # self.novel_queries = self.extract_aug_novel_proto(self.supp_dir, self.supp_path, way=len(self.novel_class), shot=self.shot)
                    print('Done! The dimension of novel prototypes is: ', self.novel_queries.shape)
                return self.forward_test(img, img_metas, self.novel_queries, **kwargs)
            else:
                return self.forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img):
        """Extract features from images."""
        visual_feat = self.backbone(img)
        return visual_feat

    def extract_novel_proto(self, dir, path, way, shot):
        ## load Image and Annotations, no augmentation but how to handle the crop??
        ## seed from GFS-Seg: 5
        all_novel_queries = np.zeros([way, 768]) # [way, dim]

        # generate label for each image
        labels = [[i]*shot for i in range(way)]
        labels = list(itertools.chain.from_iterable(labels))

        n = 0
        f = open(path, 'r')
        for filename in f:
            filename = filename.strip('\n')
            # print(filename)
            if len(self.CLASSES) == 21: ## VOC
                image_path = dir + '/JPEGImages/' + str(filename) + '.jpg'
                # print('image_path:', image_path)
                label_path = dir + '/Annotations/' + str(filename) + '.png'
                # print('label_path:', label_path)
            elif len(self.CLASSES) == 81: ## COCO2014
                image_path = dir + '/JPEGImages/train2014/' + str(filename) + '.jpg'
                label_path = dir + '/Annotations/train_contain_crowd/' + str(filename) + '.png'
            else:
                assert AttributeError('Do not support this dataset!')

            print('support_novel_image_path:', image_path)
            image_ori = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image_ori = np.float32(image_ori) # (0-255)
            label_ori = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

            ## check the image and label
            image, label = self.val_supp_transform(image_ori, label_ori)
            # image = image.unsqueeze(0).to(self.backbone.patch_embed.proj.weight.device)
            try: image = image.unsqueeze(0).to(self.backbone.class_token.device)
            except: image = image.unsqueeze(0).to(self.backbone.cls_token.device)
            # label[label==0] = 255 ## ignore the ground truth label
            # label[label!=255] -= 1
            
            # from PIL import Image
            # import matplotlib.pyplot as plt
            # plt.imshow(image.squeeze().permute(1,2,0).cpu().numpy()) / plt.imshow(label.squeeze().cpu().numpy())
            # plt.savefig('image1.png') / plt.savefig('label1.png')

            # get all patch features
            patch_embeddings = self.extract_feat(image)[0][0]  ## V1: (1, dim, 32, 32) dino+vpt better
            # patch_embeddings = self.extract_feat(image)[-1] ## V2: only from the original dino

            # obtain the mask
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int()
            binary_label = torch.zeros_like(label)
            binary_label[label == self.novel_class[labels[n]]] = 1
            assert binary_label.sum() != 0
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            proto = (torch.einsum("dhw,hw->dhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1)) / binary_label.sum()  # dim

            # print('proto:', str(int(n/num_each_supp))+str(labels[n]))
            all_novel_queries[labels[n], :] += proto.clone().cpu().numpy()
            n += 1
        
        # norm for 1shot or 5shot
        all_novel_queries /= shot

        return all_novel_queries

    def extract_aug_novel_proto(self, dir, path, way, shot):
        ## load Image and Annotations, no augmentation but how to handle the crop??
        ## seed from GFS-Seg: 5
        all_novel_queries = np.zeros([way, 768]) # [way, dim]

        # generate label for each image
        aug_num = 5
        labels = [[i]*(shot) for i in range(way)]
        labels = list(itertools.chain.from_iterable(labels))

        n = 0
        f = open(path, 'r')
        for filename in f:
            filename = filename.strip('\n')
            # print(filename)
            if len(self.CLASSES) == 21: ## VOC
                image_path = dir + '/JPEGImages/' + str(filename) + '.jpg'
                # print('image_path:', image_path)
                label_path = dir + '/Annotations/' + str(filename) + '.png'
                # print('label_path:', label_path)
            elif len(self.CLASSES) == 81: ## COCO2014
                image_path = dir + '/JPEGImages/train2014/' + str(filename) + '.jpg'
                label_path = dir + '/Annotations/train_contain_crowd/' + str(filename) + '.png'
            else:
                assert AttributeError('Do not support this dataset!')

            image_ori = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
            image_ori = np.float32(image_ori) # (0-255)
            label_ori = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) # 0 is the background

            ## check the image and label
            image, label = self.val_supp_transform(image_ori, label_ori)
            image = image.unsqueeze(0).to(self.backbone.patch_embed.proj.weight.device)
            # label[label==0] = 255 ## Do not ignore the ground truth label
            # label[label!=255] -= 1
            label = label.unsqueeze(0)

            ## how about generate augmented support images?
            for a_n in range(aug_num):
                image_aug, label_aug = self.val_supp_aug_transform(image_ori, label_ori)
                image_aug = image_aug.unsqueeze(0).to(self.backbone.patch_embed.proj.weight.device)
                # label_aug[label_aug==0] = 255 ## ignore the ground truth label
                # label_aug[label_aug!=255] -= 1
                label_aug = label_aug.unsqueeze(0)
                
                image = torch.concat((image, image_aug), dim=0)
                label = torch.concat((label, label_aug), dim=0)
            
            # from PIL import Image
            # import matplotlib.pyplot as plt
            # plt.imshow(image.squeeze().permute(1,2,0).cpu().numpy()) / plt.imshow(label.squeeze().cpu().numpy())
            # plt.savefig('image1.png') / plt.savefig('label1.png')

            # get all patch features
            patch_embeddings = self.extract_feat(image)[0][0]  ## V1: (1+aug, dim, 32, 32) dino+vpt better
            # patch_embeddings = self.extract_feat(image)[-1] ## V2: only from the original dino

            # obtain the average mask
            label = F.interpolate(label.unsqueeze(0).float(), size=patch_embeddings.shape[-2:], mode='nearest').squeeze().int() # (1+aug, 32, 32)
            binary_label = torch.zeros_like(label) # (bs, 32, 32)
            binary_label[label == self.novel_class[labels[n]]] = 1 # (bs, 32, 32)
            assert binary_label.sum() != 0
            # patch_embeddings = F.interpolate(patch_embeddings, size=binary_label.size(), mode='bilinear', align_corners=False)
            proto = torch.einsum("bdhw,bhw->bdhw", patch_embeddings.squeeze(), binary_label.to(patch_embeddings.device)).sum(-1).sum(-1) # (bs, 768)
            proto = proto / binary_label.sum(-1).sum(-1).unsqueeze(-1).to(proto.dtype).to(proto.device)  # (bs, dim) / (bs, 1)
            proto = proto.mean(dim=0) # dim

            # print('proto:', str(int(n/num_each_supp))+str(labels[n]))
            all_novel_queries[labels[n], :] += proto.clone().cpu().numpy()
            n += 1
        
        # norm for 1shot or 5shot
        all_novel_queries /= shot

        return all_novel_queries

    def extract_base_proto_epoch(self, qs, patch_features, targets):
        ## qs(base, 768), patch(bs, 768, 32, 32), gt(bs, 512, 512)
        assert patch_features.shape[0] == targets.shape[0]
        patch_features = F.interpolate(patch_features, size=targets.shape[-2:], mode='bilinear', align_corners=False) ## (512, 512)
        # targets = F.interpolate(targets.unsqueeze(1).float(), size=patch_features.shape[-2:], mode='nearest').squeeze(1).int() ## (32, 32)

        qs_epoch = torch.zeros_like(qs) # [15, dim]
        num_base = torch.zeros(qs.shape[0]).to(qs_epoch.device)  #(15)

        n = 0
        for targets_per_image in targets:
            # for n th image in a batch
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != 255]
            if len(gt_cls) != 0:
                for cls in gt_cls:
                    num_base[cls] += 1
                    binary_mask = torch.zeros_like(patch_features[0,0])
                    binary_mask[targets_per_image == cls] = 1
                    proto_cls = (torch.einsum("dhw,hw->dhw", patch_features[n].squeeze(), binary_mask).sum(-1).sum(-1)) / binary_mask.sum()
                    qs_epoch[cls, :] = proto_cls
            n += 1

        # norm for each base classes
        qs_epoch[num_base!=0] = qs_epoch[num_base!=0] / num_base[num_base!=0].unsqueeze(-1) #(15, 768)

        return qs_epoch


    def forward_train(self, img, img_metas, gt_semantic_seg):
        # print('check_weights_clip_vpt:', self.backbone.prompt_embeddings.sum())
        # print('check_weights_clip_vpt:', self.backbone.prompt_embeddings.requires_grad)
        ## check whether this image includes novel classes, gt:(bs, 1, 512, 512)
        assert gt_semantic_seg.unique() not in self.novel_class
        if len(self.base_class) != len(self.both_class): # few-shot setting
            gt_semantic_seg = torch.Tensor(self.visibility_seen_mask).type_as(gt_semantic_seg)[gt_semantic_seg]

        # print('gt:', gt_semantic_seg.unique())
        visual_feat = self.extract_feat(img) # (bs, 1025, 768)
        feat = []
        feat.append(visual_feat)
        qs_epoch = self.extract_base_proto_epoch(self.decode_head.base_qs, visual_feat[0][0].clone().detach(), gt_semantic_seg.squeeze()) # V1: from dino+vpt better
        # qs_epoch = self.extract_base_proto_epoch(self.decode_head.base_qs, visual_feat[-1], gt_semantic_seg.squeeze()) # V2: only from dino

        losses = dict()
        loss_decode = self._decode_head_forward_train(feat, img_metas, gt_semantic_seg, qs_epoch)
        losses.update(loss_decode)
        return losses
        
    def encode_decode(self, img, img_metas, novel_queries=None):
        visual_feat = self.extract_feat(img)

        feat = []
        feat.append(visual_feat)

        if novel_queries is not None:
            out = self._decode_head_forward_test(feat, img_metas, novel_queries)
        else:
            out = self._decode_head_forward_test(feat, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale, novel_protoes=None):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = len(self.both_class)
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if novel_protoes is not None:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta, novel_protoes)
                else:
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds



