python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed0.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed0 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed1.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed1 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed2.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed2 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed3.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed3 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed4.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed4 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_0/iter_5000.pth
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1__iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1__iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1__iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1__iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1__iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_5_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_0_512x512_20k_12_10_shot_5_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_0_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt


python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed0.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed0 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed1.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed1 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed2.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed2 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed3.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed3 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed4.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed4 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_1/iter_5000.pth
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_5_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_1_512x512_20k_12_10_shot_5_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_1_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt


python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed0.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed0 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed1.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed1 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed2.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed2 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed3.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed3 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed4.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed4 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_2/iter_5000.pth
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_5_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_2_512x512_20k_12_10_shot_5_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_2_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
# python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt


python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed0.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed0 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed1.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed1 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed2.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed2 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed3.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed3 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3/iter_5000.pth
python train.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed4.py --work-dir=/media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed4 --load-from=/media/data/ziqin/work_dirs_fss/voc/dino_b_16_split_3/iter_5000.pth
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_1_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt'

python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_5_seed0.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed0/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_5_seed1.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed1/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_5_seed2.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed2/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_5_seed3.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed3/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'
python test.py configs/voc12_ft_kvf/kvf_dino_b_16_split_3_512x512_20k_12_10_shot_5_seed4.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/split_3_seed4/iter_2000.pth --eval=mIoU  --savetxt='/media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt'

python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot1_iter_2000_results.txt
python cal_multiseed.py /media/data/ziqin/work_dirs_fss/voc_ft_kvf/shot5_iter_2000_results.txt