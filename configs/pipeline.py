from config import patch_size


train_pipeline = [
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(patch_size, patch_size), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            dict(type='Identify'),
            dict(type='Rotate', prob=1, min_mag=90., max_mag=90., reversal_prob=0.),
            dict(type='Rotate', prob=1, min_mag=180., max_mag=180., reversal_prob=0.),
            dict(type='Rotate', prob=1, min_mag=90., max_mag=90., reversal_prob=1.),
        ]
    ),
    dict(type='AlbumentationsTransform'),
    dict(type='PackDetInputs'),
]

valid_pipeline = [
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', scale=(patch_size, patch_size), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]
