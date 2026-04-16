auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=4,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True
)
dataset_type = 'CustomDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        type='mmengine.hooks.CheckpointHook',
        # max_keep_ckpts=30,
        # save_begin=41
    ),
    logger=dict(interval=10, type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'),
    visualization=dict(
        enable=False, type='mmpretrain.engine.hooks.VisualizationHook'))
custom_hooks = [dict(type='DumpFeatureHook',out_dir='prelogits_out')]
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
# launcher = 'pytorch'
launcher = 'none'
# load_from = None
load_from = '/home/bsj/code/mmpretrain-main/swin_large_patch4_window7_224_22kto1k-5f0996db.pth'
log_level = 'INFO'

model = dict(
    backbone=dict(
        arch='large',
        img_size=224,
        frozen_stages=3,
        # lora_cfg=dict(
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.05,
        #     enable_atten=True,
        #     enable_ffn=False,
        #     atten_linear_names=('qkv','proj'),
        #     freeze_base=True,
        #     only_lora_trainable=True,
        #     train_bias=False
        # ),
        type='mmpretrain.models.SwinTransformer'),
    head=dict(
        # cal_acc=True,
        in_channels=1536,
        # loss=dict(label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        loss=dict(loss_weight=1.0, type='mmpretrain.models.CrossEntropyLoss'),
        num_classes=4,
        topk=(
            1,
        ),
        type='mmpretrain.models.LinearClsHead'),
    neck=dict(type='mmpretrain.models.GlobalAveragePooling'),
    type='mmpretrain.models.ImageClassifier')

optim_wrapper = dict(
    clip_grad=dict(max_norm=5.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0005,   # 0.001-->0.0003
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
model_wrapper_cfg = dict(type='MMDistributedDataParallel',find_unused_parameters=True)
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.001,
        type='mmengine.optim.LinearLR'),
    dict(
        begin=5,
        by_epoch=True,
        eta_min=1e-05,
        type='mmengine.optim.CosineAnnealingLR'),
]
randomness = dict(deterministic=True, seed=220)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=48,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        # data_prefix='/home/bsj/home/bsj/datasetset/BACH_color_split/test',  # change
        data_prefix='/home/bsj/dataset/BACH_color_split/test',  # change
        with_label=True,
        pipeline=[
            dict(type='mmpretrain.datasets.LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='mmpretrain.datasets.ResizeEdge'),
            dict(crop_size=224, type='mmpretrain.datasets.CenterCrop'),
            dict(type='mmpretrain.datasets.PackInputs'),
        ],
        # split='val',
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
test_evaluator = [
    dict(topk=(1,), type='Accuracy'),
    dict(type='SingleLabelMetric', items=['precision','recall','f1-score','support']),
    dict(type='ConfusionMatrix'),
    dict(type='SingleLabelAUC')
]
test_pipeline = [
    dict(type='mmpretrain.datasets.LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='mmpretrain.datasets.ResizeEdge'),
    dict(crop_size=224, type='mmpretrain.datasets.CenterCrop'),
    dict(type='mmpretrain.datasets.PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=70, val_interval=200)
train_dataloader = dict(
    batch_size=48,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        # data_prefix='/home/bsj/home/bsj/datasetset/BACH_color_split/train',  # change
        data_prefix='/home/bsj/dataset/BACH_color_split/train',  # change
        with_label=True,
        pipeline=[
            dict(type='mmpretrain.datasets.LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='mmpretrain.datasets.RandomResizedCrop'),
            dict(
                direction='horizontal',
                prob=0.5,
                type='mmpretrain.datasets.RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='mmpretrain.datasets.RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='mmpretrain.datasets.RandomErasing'),
            dict(type='mmpretrain.datasets.PackInputs'),
        ],
        # split='train',
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
train_pipeline = [
    dict(type='mmpretrain.datasets.LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='mmpretrain.datasets.RandomResizedCrop'),
    dict(
        direction='horizontal',
        prob=0.5,
        type='mmpretrain.datasets.RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='mmpretrain.datasets.RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='mmpretrain.datasets.RandomErasing'),
    dict(type='mmpretrain.datasets.PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=48,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        # data_prefix='/home/bsj/home/bsj/datasetset/BACH_color_split/test',  # change
        data_prefix='/home/bsj/dataset/BACH_color_split/test',  # change
        with_label=True,
        pipeline=[
            dict(type='mmpretrain.datasets.LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='mmpretrain.datasets.ResizeEdge'),
            dict(crop_size=224, type='mmpretrain.datasets.CenterCrop'),
            dict(type='mmpretrain.datasets.PackInputs'),
        ],
        # split='val',
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='mmengine.dataset.DefaultSampler'))
val_evaluator  = [
    dict(topk=(1,), type='Accuracy'),
    dict(type='SingleLabelMetric', items=['precision','recall','f1-score','support']),
    dict(type='ConfusionMatrix'),
    dict(type='SingleLabelAUC')
]
vis_backends = [
    dict(type='mmengine.visualization.LocalVisBackend'),
]
visualizer = dict(
    type='mmpretrain.visualization.UniversalVisualizer',
    vis_backends=[
        dict(type='mmengine.visualization.LocalVisBackend'),
    ])
