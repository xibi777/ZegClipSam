
## gt
# CUDA_VISIBLE_DEVICES="1" python test.py configs/visualization/coco_vit_split_3_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/coco_vit/vit_b_16_split_3/iter_40000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/gt_coco_split_3/
# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/voc_dino_split_0_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc12_dino/dino_b_16_split_3/iter_10000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/gt_voc_split_0/

# ## DINO VS ViT
# python test.py configs/visualization/voc_vit_split_0_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc12_vit/vit_b_16_split_0/iter_10000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/pred_voc_vit_split_0/
# python test.py configs/visualization/voc_dino_split_0_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc12_dino/dino_b_16_split_0/iter_10000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/pred_voc_dino_split_0/

# CUDA_VISIBLE_DEVICES="1" python test.py configs/visualization/coco_dino_split_3_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/coco_dino/dino_b_16_split_3/iter_40000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/pred_coco_dino_split_3/
# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/coco_vit_split_3_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/coco_vit/vit_b_16_split_3/iter_40000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization/pred_coco_vit_split_3/



# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/binary_voc_dino_split_0_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc12_dino/dino_b_16_split_0/iter_10000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization_binary/pred_voc_dino_split_0/
# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/binary_voc_vit_split_0_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc12_vit/vit_b_16_split_0/iter_10000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization_binary/pred_voc_vit_split_0/

# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/binary_coco_dino_split_3_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/coco_dino/dino_b_16_split_3/iter_40000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization_binary/pred_coco_dino_split_3/
# CUDA_VISIBLE_DEVICES="3" python test.py configs/visualization/binary_coco_vit_split_3_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/coco_vit/vit_b_16_split_3/iter_40000.pth --show --show-dir=/media/data/ziqin/work_dirs_fss/visualization_binary/pred_coco_vit_split_3/


CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_EL_RD/dino_b_16_fully_512x512_20k_fix.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_EL_RD/iter_10000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_EL_WORD/dino_b_16_fully_512x512_20k_fix.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_EL_WORD/iter_10000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_NEL_WORD/dino_b_16_fully_512x512_20k_fix.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/Fix_NEL_WORD/iter_10000.pth --eval=mIoU

CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/VPT_EL_RD/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/VPT_EL_RD/iter_10000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/VPT_EL_WORD/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc12_dino_fully/VPT_EL_WORD/iter_10000.pth --eval=mIoU

