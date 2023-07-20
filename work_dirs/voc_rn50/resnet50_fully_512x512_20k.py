norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 2048
out_indices = [11]
model = dict(
    type='FewSegViT',
    pretrained='/media/data/ziqin/pretrained/resnet/resnet50-19c8e357.pth',
    context_length=77,
    backbone=dict(
        type='LoRAResNet',
        layers=[3, 4, 6, 3],
        style='pytorch',
        pretrained='/media/data/ziqin/pretrained/resnet/resnet50-19c8e357.pth'
    ),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=512,
        in_channels=2048,
        channels=512,
        num_classes=21,
        num_layers=3,
        num_heads=8,
        use_stages=1,
        embed_dims=512,
        loss_decode=dict(
            type='SegLossPlus',
            num_classes=21,
            dec_layers=3,
            loss_weight=1.0,
            mask_weight=20.0,
            dice_weight=1.0),
        seen_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
        all_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ],
        use_proj=True),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)),
    base_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ],
    novel_class=[],
    both_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ],
    split=0,
    shot=1,
    supp_dir=None,
    supp_path=None,
    ft_backbone=False,
    exclude_key='lora')
dataset_type = 'ZeroPascalVOCDataset21'
data_root = '/media/data/ziqin/data_fss/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
split = 0
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ZeroPascalVOCDataset21',
        data_root='/media/data/ziqin/data_fss/VOC2012',
        img_dir='JPEGImages',
        ann_dir='Annotations',
        split='ImageSets/ShotSegmentation/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='ZeroPascalVOCDataset21',
        data_root='/media/data/ziqin/data_fss/VOC2012',
        img_dir='JPEGImages',
        ann_dir='Annotations',
        split='ImageSets/ShotSegmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ZeroPascalVOCDataset21',
        data_root='/media/data/ziqin/data_fss/VOC2012',
        img_dir='JPEGImages',
        ann_dir='Annotations',
        split='ImageSets/ShotSegmentation/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True, min_size=512),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(
    type='AdamW',
    lr=2e-05,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=1.0),
            norm=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            head=dict(lr_mult=1.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=10001, metric='mIoU')
channels = 512
base_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
novel_class = []
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
num_classes = 21
pretrained = '/media/data/ziqin/pretrained/resnet/resnet50-19c8e357.pth'
work_dir = './work_dirs/voc_rn50'
gpu_ids = range(0, 1)
