## Environment:
-Install pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch

-Install the mmsegmentation library and some required packages.
pip install mmcv-full==1.4.4 mmsegmentation==0.24.0
pip install scipy timm==0.3.2

## Preparing Dataset:
According to MMseg: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md

## Preparing Pretrained CLIP model:
Download the pretrained model here: ./pretrained/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## Training (Inductive):
bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_plus_multi.py ./work_dirs/coco/zero_12_100
bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10_plus_multi.py ./work_dirs/voc12/zero_12_10

## Training (Transductive):
bash dist_train.sh ./configs/coco/vpt_seg_zero_vit-b_512x512_40k_12_100_plus_multi_st.py ./work_dirs/coco/zero_12_100_st --load-from=./work_dirs/coco/zero_12_100/iter_40000.pth
bash dist_train.sh ./configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_plus_multi_st.py ./work_dirs/voc12/zero_12_10_st --load-from=./work_dirs/voc12/zero_12_10/iter_10000.pth

## Training (Fully supervised):
bash dist_train.sh configs/coco/vpt_seg_fully_vit-b_512x512_80k_12_100_plus_multi.py ./work_dirs/coco/fully_12_100
bash dist_train.sh configs/voc12/vpt_seg_fully_vit-b_512x512_20k_12_10_plus_multi.py ./work_dirs/voc12/fully_12_10

## Testing:
CUDA_VISIBLE_DEVICES="0" python test.py ./path/to/config ./path/to/model.pth --eval=mIoU

For example: CUDA_VISIBLE_DEVICES="0" python test.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_plus_multi.py work_dirs/coco/zero_12_100/latest.pth --eval=mIoU


## Cross Dataset Inference:
## COCO->
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-voc.py ./path/to/coco/vpt_seg_zero_80k_12_100_plus_multi/iter_80000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-context.py ./path/to/coco/vpt_seg_zero_80k_12_100_plus_multi/iter_80000.pth --eval=mIoU

## Context->
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/context-to-voc.py ./path/to/context/vpt_seg_zero_40k_12_35_plus_multi/iter_40000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/context-to-coco.py ./path/to/context/vpt_seg_zero_40k_12_35_plus_multi/iter_40000.pth --eval=mIoU