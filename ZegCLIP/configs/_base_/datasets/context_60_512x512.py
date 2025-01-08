dataset_type = 'ZeroPascalContextDataset60'
data_root = '/media/data/ziqin/data/context/VOCdevkit/VOC2010/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (520, 520)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='DefaultFormatBundle_Mask'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'mask']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadMaskFromFile'),
    # dict(type='LoadAnnotations'),  ##ADDED
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, min_size=512),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='ImageToTensor', keys=['img']),
            # dict(type='Collect', keys=['img', 'gt_semantic_seg']), ##ADDED
            dict(type='ImageToTensor', keys=['img', 'mask']),
            dict(type='Collect', keys=['img', 'mask']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClassContext',
        split='ImageSets/SegmentationContext/val.txt',
        pipeline=test_pipeline))