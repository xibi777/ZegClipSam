_base_ = [
    '../_base_/models/fewsegvit.py', '../_base_/datasets/finetuning_config/voc12_512x512_shot_1_seed0.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_200.py'
]

img_size = 512
in_channels = 768 # 512?
out_indices = [11]

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
novel_class = [11, 12, 13, 14, 15]
both_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
num_classes = len(base_class)

eval_supp_dir = '/media/data/ziqin/data_fss/VOC2012'
eval_supp_path = '/media/data/ziqin/data_fss/VOC2012/ImageSets/FewShotSegmentation/val_supp_split_2_shot_1_seed0.txt'

pretrained = '/media/data/ziqin/pretrained/dino_vitbase16_pretrain.pth'

model = dict(
    type='FewSegViT',
    pretrained=pretrained, 
    context_length=77,
    backbone=dict(
        type='PromptVisionTransformer',
        img_size = 512,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        out_indices=out_indices, 
        pretrained=pretrained, 
        #setting of vpt
        num_tokens=10,
        prompt_dim=768,
        total_d_layer=11,
        style='pytorch'),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=3,
        num_heads=8,
        use_proj=False,
        cls_type='weighted',
        backbone_type='dino',
        use_stages=len(out_indices),
        out_indices=out_indices,
        embed_dims=in_channels,
        finetune=True,
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
    split = 2,
    shot = 1,
    supp_dir = eval_supp_dir,
    supp_path = eval_supp_path,
    ft_backbone = False,
    exclude_key='prompt',
    finetune=True,
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=1.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=1.),
                                        }))

data = dict(samples_per_gpu=4,
            workers_per_gpu=4,)
