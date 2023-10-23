norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [11]
model = dict(
    type='CAPLSegViT',
    pretrained='/media/data/ziqin/pretrained/B_16.pth',
    context_length=77,
    backbone=dict(
        type='BaseImageNetViT',
        layers=[3, 4, 6, 3],
        style='pytorch',
        out_indices=[11],
        pretrained='/media/data/ziqin/pretrained/B_16.pth'),
    decode_head=dict(
        type='CAPLHeadSeg',
        img_size=512,
        in_channels=768,
        num_classes=15,
        loss_decode=dict(
            type='CELoss',
            num_classes=16,
            dec_layers=3,
            loss_weight=1.0,
            ce_weight=1.0),
        channels=768,
        seen_idx=[0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        all_idx=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)),
    base_class=[0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    novel_class=[1, 2, 3, 4, 5],
    both_class=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20
    ],
    split=0,
    shot=1,
    supp_dir='/media/data/ziqin/data_fss/VOC2012',
    supp_path=
    '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/val_supp_split_0_shot_1.txt',
    ft_backbone=False)
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
    samples_per_gpu=4,
    workers_per_gpu=4,
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
    type='SGD',
    lr=0.001,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=1.0),
            norm=dict(decay_mult=0.0),
            ln=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=20001, metric='mIoU')
base_class = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
novel_class = [1, 2, 3, 4, 5]
both_class = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
]
num_classes = 16
eval_supp_dir = '/media/data/ziqin/data_fss/VOC2012'
eval_supp_path = '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/val_supp_split_0_shot_1.txt'
pretrained = '/media/data/ziqin/pretrained/B_16.pth'
work_dir = './work_dirs_fss/voc_capl/vit_b_16_split_0'
gpu_ids = range(0, 1)
