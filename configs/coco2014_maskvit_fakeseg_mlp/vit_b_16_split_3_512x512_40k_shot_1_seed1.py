_base_ = [
    '../_base_/models/fewsegvit.py', '../_base_/datasets/coco2014_512x512_split_3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

img_size = 512
in_channels = 768 # 512?
out_indices = [11]

base_class = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 
              29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 53, 54, 
              55, 57, 58, 59, 61, 62, 63, 65, 66, 67, 69, 70, 71, 73, 74, 75, 77, 78, 79] # 0:background
novel_class = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

num_classes = len(base_class)

eval_supp_dir = '/media/data/ziqin/data_fss/coco2014'
eval_supp_path = '/media/data/ziqin/data_fss/coco2014/ImageSets/FewShotSegmentation/val_supp_split_3_shot_1_seed1.txt'

pretrained = '/media/data/ziqin/pretrained/B_16.pth'

model = dict(
    type='MaskFakeFewSegViT',
    pretrained=pretrained, 
    context_length=77,
    backbone=dict(
        type='MaskPromptImageNetViT',
        ## ADDED
        out_indices=out_indices, 
        pretrained=pretrained, 
        #setting of vpt
        num_tokens=50,
        prompt_dim=768,
        total_d_layer=11,
        style='pytorch'),
    decode_head=dict(
        type='MaskFakeHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=1,
        num_heads=1,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=in_channels,
        rd_type='qclsq',
        decode_type='mlp',
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3, 
            mask_weight=20.0, #20.0
            dice_weight=1.0,
            loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)), 
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    split = 3,
    shot = 1,
    supp_dir = eval_supp_dir,
    supp_path = eval_supp_path,
    ft_backbone = False,
    exclude_key='prompt',
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        }))

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,)
