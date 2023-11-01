# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import torch
import torch.nn.functional as F
from torch import nn
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import torch.distributed as dist

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def cosine_margin_loss(q, e, labels, tau=1.0, m=0.5):
    # q:(bs, n_cls, 512) - (bs*n_cls, 512), e:(n_cls+1, 512), labels:(bs*n_cls), the value is from 0 to n_cls(n_cls+1, -1 is learnanle)
    assert q.shape[1]+1 == e.shape[0]
    bs, n_cls, n_dim = q.shape
    q = q.reshape(bs*n_cls, n_dim)
    ## do I need to norm q and e before?
    pos = torch.exp(F.cosine_similarity(q, e[labels.long()].reshape(bs*n_cls, n_dim)) / tau) # [bs*n_cls]
    neg = torch.exp(F.cosine_similarity(q.unsqueeze(1), e.unsqueeze(0), dim=-1) / tau) # [bs*n_cls, n_cls+1]
    neg = torch.sum(neg, dim=-1) + m #[bs*n_cls]
    return 1 - torch.mean(torch.div(pos, neg))


class SegPlusCriterion(nn.Module):
    # in this version, both all masks and logits will be added to compute loss
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, losses, eos_coef=0.1):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        outputs: pred_logits: (bs, n_cls, 1)                       targets: len = bs
                 pred_masks:  (bs, n_cls, H, W)                    targets[0]: 'labels': eg: have the [2, 4] th classes = 2
                 pred: (bs, n_cls, H, W) = pred_logits*pred_masks              'masks':  eg: (2, H, W)
                 aux_outputs: mediate outputs
        """
        assert "pred_masks" in outputs

        # for focal loss
        src_masks = outputs["pred_masks"] # (bs, cls, H, W)
        target_masks = self._get_target_mask_binary_cross_entropy(src_masks, targets)

        bs, n_cls, H, W = target_masks.size()
        _, _, H_, W_ = src_masks.size()
        src_masks = src_masks.reshape(bs*n_cls, H_, W_)
        target_masks = target_masks.reshape(bs*n_cls, H, W)
        # upsample predictions to the target size
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        # for dice loss
        src_idx = self._get_src_permutation_idx(indices) #dim0: the class number belongs to each image; dim1: each class number of each image
        tgt_idx = self._get_tgt_permutation_idx(indices) #dim0: the class number belongs to each image; dim1: each class number (sorted from 0) of each image
        # eg: src_idx: [0, 1, 1, 2] [1, 2, 13, 0] 
        # eg: tgt_idx: [0, 1, 1, 2] [0, 0, 1, 0] 
        src_masks_dice = outputs["pred_masks"] # (bs, cls, H, W)
        if src_masks_dice.dim() != 4:
            return {"no_loss": 0}
        src_masks_dice = src_masks_dice[src_idx] # (num_mask, H, W)
        masks_dice = [t["target_masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks_dice, valid = nested_tensor_from_tensor_list(masks_dice).decompose()
        target_masks_dice = target_masks_dice.to(src_masks_dice)
        target_masks_dice = target_masks_dice[tgt_idx]

        # upsample predictions to the target size --> for aug_loss
        src_masks_dice = F.interpolate(
            src_masks_dice[:, None], size=target_masks_dice.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks_dice = src_masks_dice[:, 0].flatten(1)

        target_masks_dice = target_masks_dice.flatten(1)
        target_masks_dice = target_masks_dice.view(src_masks_dice.shape)
        
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks_dice, target_masks_dice, num_masks),
        }
        return losses

    def _get_target_mask_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2]
        H, W = targets[0]['masks'].size()
        target_masks_o = torch.zeros(B, C, H*W).to(out_masks.device) # (bs, cls, H*W)
        for i, target in enumerate(targets):
            mask = target['masks'].long().reshape(-1) # (H*W)
            idx = torch.arange(0, H*W, 1).long().to(out_masks.device) # (H*W)
            mask_o = mask[mask!=255]
            idx = idx[mask!=255]
            target_masks_o[i, mask_o, idx] = 1
        return target_masks_o.reshape(B, C, H, W)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets

        labels = [x['labels'] for x in targets] # labels of each sample in a batch, each label contains the class of each image
        indices_new = []
        for label in labels:
            t = torch.arange(len(label))
            indices_new.append([label, t])
        indices = indices_new
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        ) # total number of the whole classes
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses, loss_label and loss_mask
        losses = {}
        for loss in self.losses: #('labels and masks)
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # use the indices as the last stage
                aux_loss = ["masks"]
                for loss in aux_loss:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

class CECriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, losses, eos_coef=0.1):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_masks_ce(self, outputs, targets):
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"] #(bs, 150, 512, 512)
        target_masks = torch.concat([x['masks'].unsqueeze(0) for x in targets]) #(bs, 512, 512)

        src_masks = F.interpolate(
            src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )

        ## F.cross_entropy has been used softmax
        # losses = {
        #     "loss_mask_ce": F.cross_entropy(src_masks, target_masks, ignore_index=255),
        # }

        ## softmax/tau + crossentropy
        tau = 1
        src_masks = F.log_softmax(src_masks/tau, dim=1)

        losses = {
            "loss_mask_ce": F.nll_loss(src_masks, target_masks, ignore_index=255),
        }

        
        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {"masks_ce": self.loss_masks_ce}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):   
        # Compute all the requested losses, loss_label and loss_mask
        losses = {}
        for loss in self.losses: #('labels and masks)
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # use the indices as the last stage
                aux_loss = ["masks_ce"]
                for loss in aux_loss:
                    l_dict = self.get_loss(loss, aux_outputs, targets)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
