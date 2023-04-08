# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512
in_channels = 768
out_indices = [5, 7, 11]

model = dict(
    type='FewSegViT',
    pretrained='/media/data/ziqin/pretrained/RN50.pt',
    context_length=5,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        style='pytorch'),
    decode_head=dict(
        type='CAPLSegHead',
        img_size=img_size,
        in_channels=in_channels,
        num_classes=15,
        loss_decode=dict(
            type='SegPlussLoss',  num_classes=15, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
    )
