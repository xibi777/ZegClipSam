from models.losses.criterion import CECriterion

from models.segmentor.fewsegvit import FewSegViT
from models.segmentor.ft_fewsegvit import FTFewSegViT
from models.segmentor.freeseg import FreeSeg
from models.segmentor.binary_fewsegvit import BinaryFewSegViT
from models.segmentor.binary_fewsegvit_fake import BinaryFakeFewSegViT
from models.segmentor.cross_fewsegvit import CrossFewSegViT
from models.segmentor.caplsegvit import CAPLSegViT
from models.segmentor.fewsegvit_fake import FakeFewSegViT, MaskFakeFewSegViT

from models.backbone.dino_encoder import BaseVisionTransformer, PromptVisionTransformer
from models.backbone.mae_encoder import PromptMaskedAutoencoderViT
from models.backbone.beit_encoder import PromptBEiT
from models.backbone.vit_encoder import PromptImageNetViT, BaseImageNetViT, MaskPromptImageNetViT
from models.backbone.resnet_encoder import LoRAResNet, MyResNet

from models.decode_heads.decode_seg import ATMSingleHeadSeg, ATMSingleHeadSegWORD
from models.decode_heads.decode_seg_fake import FakeHeadSeg, BinaryFakeHeadSeg
from models.decode_heads.ft_decode_seg import FTATMSingleHeadSeg
from models.decode_heads.free_decode_seg import FreeHeadSeg
from models.decode_heads.cross_decode_seg import CrossATMSingleHeadSeg
from models.decode_heads.capl_decoder_seg import CAPLHeadSeg
from models.decode_heads.psp_decode_seg import PSPHeadSeg

from models.losses.atm_loss import SegLossPlus, FCLoss, CECriterion

from configs._base_.datasets.dataloader.voc12_21 import ZeroPascalVOCDataset21
from configs._base_.datasets.dataloader.voc12_binary import BinaryPascalVOCDataset20
from configs._base_.datasets.dataloader.coco2014_81 import ZeroCOCO2014Dataset81
from configs._base_.datasets.dataloader.coco2014_binary import BinaryCOCO2014Dataset
