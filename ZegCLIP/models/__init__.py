from models.segmentor.zegclip import ZegCLIP

from models.backbone.text_encoder import CLIPTextEncoder
from models.backbone.img_encoder import CLIPVisionTransformer, VPTCLIPVisionTransformer
from models.decode_heads.decode_seg import ATMSingleHeadSeg

from models.losses.atm_loss import SegLossPlus

from configs._base_.datasets.dataloader.voc12 import ZeroPascalVOCDataset20
from configs._base_.datasets.dataloader.coco_stuff import ZeroCOCOStuffDataset
from configs._base_.datasets.dataloader.context60 import ZeroPascalContextDataset60
from configs._base_.datasets.dataloader.ade20k_847 import MyADE20KDataset847

from configs._base_.datasets.pipeline.loading_mask import LoadMaskFromFile
from configs._base_.datasets.pipeline.transform_mask import DefaultFormatBundle_Mask
