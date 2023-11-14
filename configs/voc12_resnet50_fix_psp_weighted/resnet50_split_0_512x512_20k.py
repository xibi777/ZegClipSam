_base_ = [
    '../_base_/models/fewsegvit.py', '../_base_/datasets/voc12_512x512_fully.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

img_size = 512
in_channels = 2048
channels = 512

base_class = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
novel_class = [1, 2, 3, 4, 5]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
num_classes = len(base_class)

eval_supp_dir = '/media/data/ziqin/data_fss/VOC2012'
eval_supp_path = '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/val_supp_split_0_shot_1.txt'

pretrained = '/media/data/ziqin/pretrained/resnet/resnet50-19c8e357.pth'

model = dict(
    type='FewSegViT',
    pretrained=pretrained, 
    context_length=77,
    backbone=dict(
        type='MyResNet',
        layers=[3, 4, 6, 3],
        pretrained=pretrained, 
        style='pytorch'),
    decode_head=dict(
        type='PSPHeadSeg',
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
        use_stages=1,
        out_indices=[11],
        embed_dims=channels,
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
    supp_dir = eval_supp_dir,
    supp_path = eval_supp_path,
    ft_backbone = False,
    # exclude_key='lora',
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1e-5)


# optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01, 
#         paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
#                                         'norm': dict(decay_mult=0.),
#                                         'ln': dict(decay_mult=0.),
#                                         'head': dict(lr_mult=10.),
                                        # }))

optimizer = dict(type='SGD', lr=0.00025, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=1.),
                                        }))

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,)

