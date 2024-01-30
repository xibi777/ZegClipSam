_base_ = [
    '../_base_/models/fewsegvit.py', '../_base_/datasets/voc12_512x512_split_0.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_10k.py'
]

img_size = 512
in_channels = 1024
out_indices = [2, 3] ## the last block

base_class = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
novel_class = [1, 2, 3, 4, 5]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
num_classes = len(base_class)

eval_supp_dir = '/media/data/ziqin/data_fss/VOC2012'
eval_supp_path = '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/val_supp_split_0_shot_5_seed4.txt'

pretrained = '/media/data/ziqin/pretrained/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'

model = dict(
    type='SwinFewSegViT',
    pretrained=pretrained, 
    context_length=77,
    backbone=dict(
        type='BaseSwinTransformer',
        pretrained=pretrained, 
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        out_indices=out_indices,
        window_size=12),
    decode_head=dict(
        type='MultiATMSingleHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=3,
        num_heads=8,
        use_proj=True,
        cls_type='weighted',
        backbone_type='swin',
        use_stages=len(out_indices),
        out_indices=out_indices,
        embed_dims=in_channels,
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3, 
            mask_weight=20.0,
            dice_weight=1.0,
            loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)), 
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    split = 0,
    shot = 5,
    supp_dir = eval_supp_dir,
    supp_path = eval_supp_path,
    ft_backbone = False,
    exclude_key='tune_block',
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

