_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 1
mix_prob = 0 ## no mix
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=3,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1-2d",
        in_channels=2,
        order=("z", "z-trans"),
        stride=(2, 2, 2),
        enc_depths=(2, 2, 2, 4),
        enc_channels=(32, 64, 128, 256),
        enc_num_head=(2, 4, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2),
        dec_channels=(64, 64, 128),
        dec_num_head=(4, 4, 8),
        dec_patch_size=(1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 100
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "Icesat2Dataset"
# data_root = "/home/yanai/OhlwilerDataset2D_Seg"
data_root = "/mnt/e/Projects/ICESAT-2_Bathymetry/data_8192/pointcept_OhlwilerDataset2D_Seg"

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=[
        "sea surface",
        "sea floor",
        "others"
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            # dict(
            #     type = 'CenterShift2D',
            #     apply_y = False,
            # ),
            dict(
                type = 'PointFilter2D',
                point_cloud_range=(0, -50, 1e10, 10),
            ),
            dict(
              type = 'RandomFlipIcesat' ,
              p = 0.5,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name", "feat"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            # dict(
            #     type = 'CenterShift2D',
            #     apply_y = False,
            # ),
            dict(
                type = 'PointFilter2D',
                point_cloud_range=(0, -50, 1e10, 10),
            ),
            dict(
                type='RandomFlipIcesat',
                p=0.5,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name", "feat"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(
                type = 'PointFilter2D',
                point_cloud_range=(0, -50, 1e10, 10),
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name", "feat"),
            ),
        ],
        test_mode=True,

    ),
)
