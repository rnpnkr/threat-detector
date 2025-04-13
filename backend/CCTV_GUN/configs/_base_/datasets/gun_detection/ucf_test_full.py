# test on the entire dataset
train_dataset_type = "GunDataset"
val_dataset_type = "GunDatasetHOI"

data_root = "data/"


img_norm_cfg = dict(
    mean=[106.15, 103.56, 101.75], std=[71.933, 72.015, 72.234], to_rgb=True
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 2,

    test = dict(
        type = val_dataset_type,
        data_root = data_root,
        ann_file = "ucf/annotations_all.json",
        img_prefix = "all_images",
        pipeline = test_pipeline
    ),
    val = dict(
        type = val_dataset_type,
        data_root = data_root,
        ann_file = "ucf/annotations_all.json",
        img_prefix = "all_images",
        pipeline = test_pipeline
    )
)