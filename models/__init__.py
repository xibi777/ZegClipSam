from models.losses.criterion import CECriterion

from models.segmentor.fewsegvit import FewSegViT

from models.backbone.dino_encoder import BaseVisionTransformer, PromptVisionTransformer
from models.backbone.vit_encoder import PromptImageNetViT, BaseImageNetViT, MaskPromptImageNetViT

from models.decode_heads.decode_seg import ATMSingleHeadSeg, MultiATMSingleHeadSeg
from models.decode_heads.cross_decode_seg import CrossATMSingleHeadSeg
from models.decode_heads.decode_seg_new import PlusHeadSeg, PlusHeadSegOnlyRaw

from models.losses.atm_loss import SegLossPlus, FCLoss, CECriterion, MultiSegLossPlus

from configs._base_.datasets.dataloader.voc12_21 import ZeroPascalVOCDataset21
from configs._base_.datasets.dataloader.coco2014_81 import ZeroCOCO2014Dataset81

from configs._base_.datasets.pipeline.loading_mask import LoadMaskFromFile
from configs._base_.datasets.pipeline.transform_mask import DefaultFormatBundle_Mask
