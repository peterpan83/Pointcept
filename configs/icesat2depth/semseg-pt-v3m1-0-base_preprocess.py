_base_ = ["../_base_/default_runtime.py"]

# misc custom setting

# dataset settings
dataset_type = "Icesat2Dataset"
# data_root = "/mnt/e/Projects/ICESAT-2_Bathymetry/OhlwilerDataset2D"
data_root = "/mnt/e/Projects/ICESAT-2_Bathymetry/data_8192/data_8192/pointcept_format"

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
                type="GridSample",
                grid_size=(0.2, 0.05), ### 0.2m for along distance and 0.05m for height
                keys=('coord','height','segment'),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_count = True,
                keep_absolute_axis = [1], ## keep absolute position for y (height)
            ),
            dict(
                type = 'SegmentGrid2D',
                x_max_grid=2**16-1, ## the maximum serilization depth is 16
                x_overlap=2000,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment","name"),
                feat_keys=("height", "count"),
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
                type='PointFilter2D',
                point_cloud_range=(0, -50, 1e10, 10),
            ),
            dict(
                type="GridSample",
                grid_size=(0.2, 0.05),  ### 0.2m for along distance and 0.05m for height
                keys=('coord', 'height', 'segment'),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_count=True,
                keep_absolute_axis=[1],  ## keep absolute position for y (height)
            ),
            dict(
                type='SegmentGrid2D',
                x_max_grid=2 ** 16 - 1,  ## the maximum serilization depth is 16
                x_overlap=2000,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("height", "count"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
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
                type='PointFilter2D',
                point_cloud_range=(0, -50, 1e10, 10),
            ),
            dict(
                type="GridSample",
                grid_size=(0.2, 0.05),  ### 0.2m for along distance and 0.05m for height
                keys=('coord', 'height', 'segment'),
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_count=True,
                keep_absolute_axis=[1],  ## keep absolute position for y (height)
            ),
            dict(
                type='SegmentGrid2D',
                x_max_grid=2 ** 16 - 1,  ## the maximum serilization depth is 16
                x_overlap=2000,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "name"),
                feat_keys=("height", "count"),
            ),
        ],
        test_mode=False,

    ),
)
