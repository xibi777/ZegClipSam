_base_ = [
    '../_base_/models/fewsegvit.py', '../_base_/datasets/coco2014_512x512_fully.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

img_size = 512
in_channels = 2048 # 512?
out_indices = [11]

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
             19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 
             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 
             70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
novel_class = []
both_class = base_class
num_classes = len(base_class)

pretrained = '/media/data/ziqin/pretrained/resnet/resnet50-19c8e357.pth'

model = dict(
    type='FewSegViTSave',
    pretrained=pretrained, 
    context_length=77,
    backbone=dict(
        type='MyResNet',
        layers=[3, 4, 6, 3],
        pretrained=pretrained, 
        style='pytorch'),
    decode_head=dict(
        type='SaveHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
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
    shot = 1,
    supp_dir = None,
    supp_path = None,
    ft_backbone = False,
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

data = dict(samples_per_gpu=6,
            workers_per_gpu=6,)

