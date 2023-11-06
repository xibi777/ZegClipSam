# bash dist_train_0123.sh configs/voc12/dino_b_16_split_0_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_1_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_2_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_3_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3
# bash dist_train_0123.sh configs/voc12/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_16_fully_vpt

# python test.py configs/voc12/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_16_fully/iter_20000.pth --eval mIoU

# python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_shot_1/iter_20000.pth --eval mIoU
# python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_shot_1/iter_20000.pth --eval mIoU
# python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_shot_1/iter_20000.pth --eval mIoU
# python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_shot_1/iter_20000.pth --eval mIoU
# python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_shot_1/iter_20000.pth --eval mIoU

# CUDA_VISIBLE_DEVICES="0" python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_20000.pth --eval mIoU
# CUDA_VISIBLE_DEVICES="1" python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_20000.pth --eval mIoU
# CUDA_VISIBLE_DEVICES="2" python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_20000.pth --eval mIoU
# CUDA_VISIBLE_DEVICES="3" python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_20000.pth --eval mIoU
# CUDA_VISIBLE_DEVICES="0" python test.py configs/voc12/dino_b_16_split_0_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_20000.pth --eval mIoU

# CUDA_VISIBLE_DEVICES="3" python test.py /media/data/ziqin/work_dirs_fss/voc/dino_16_fully_vpt/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_16_fully_vpt/iter_20000.pth --eval mIoU

# bash dist_train_0123.sh configs/voc12/dino_b_16_split_0_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0_V1
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_1_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1_V1
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_2_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2_V1
# bash dist_train_0123.sh configs/voc12/dino_b_16_split_3_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3_V1
# bash dist_train_0123.sh configs/voc12/dino_b_16_fully_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc/dino_16_fully_vpt_V1

# bash dist_train_012.sh configs/voc12_resnet50_psp/resnet50_split_0_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_resnet50_psp/vit_b_16_split_0
# bash dist_train_012.sh configs/voc12_resnet50_fix_psp/resnet50_split_0_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_resnet50_fix_psp/vit_b_16_split_0
# bash dist_train_012.sh configs/voc12_resnet50_all/resnet50_split_2_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_resnet50/vit_b_16_split_2
# bash dist_train_012.sh configs/voc12_resnet50_all/resnet50_split_3_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_resnet50/vit_b_16_split_3

# bash dist_train_012.sh configs/voc12_resnet50_fakeseg_mlp_all/resnet50_split_0_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_resnet50_fakeseg_mlp/vit_b_16_split_0

# bash dist_train_012.sh configs/voc12_capl_fix/vit_b_16_split_0_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_capl_fix/vit_b_16_split_0
# bash dist_train_012.sh configs/voc12_capl_fix/vit_b_16_split_1_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_capl_fix/vit_b_16_split_1
# bash dist_train_012.sh configs/voc12_capl_fix/vit_b_16_split_2_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_capl_fix/vit_b_16_split_2
# bash dist_train_012.sh configs/voc12_capl_fix/vit_b_16_split_3_512x512_20k.py /media/data/ziqin/work_dirs_fss/voc_capl_fix/vit_b_16_split_3

# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_0_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc_plus/vit_b_16_split_0
# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_0_512x512_20k_12_10_onlyraw.py /media/data/ziqin/work_dirs_fss/voc_plus_onlyraw/vit_b_16_split_0

# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_1_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc_plus/vit_b_16_split_1
# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_1_512x512_20k_12_10_onlyraw.py /media/data/ziqin/work_dirs_fss/voc_plus_onlyraw/vit_b_16_split_1

# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_2_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc_plus/vit_b_16_split_2
# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_2_512x512_20k_12_10_onlyraw.py /media/data/ziqin/work_dirs_fss/voc_plus_onlyraw/vit_b_16_split_2

# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_3_512x512_20k_12_10.py /media/data/ziqin/work_dirs_fss/voc_plus/vit_b_16_split_3
# bash dist_train_012.sh configs/voc12_vit_all_plus/vit_b_16_split_3_512x512_20k_12_10_onlyraw.py /media/data/ziqin/work_dirs_fss/voc_plus_onlyraw/vit_b_16_split_3

bash dist_train_012.sh configs/coco_voc_save/voc_dino.py /media/data/ziqin/work_dirs_fss/save_proto/voc_dino
bash dist_train_012.sh configs/coco_voc_save/coco_dino.py /media/data/ziqin/work_dirs_fss/save_proto/coco_dino