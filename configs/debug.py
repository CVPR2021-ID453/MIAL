_base_ = [
    './_base_/retinanet_r50_fpn.py', './_base_/voc0712.py',
    './_base_/default_runtime.py'
]

data_root = '/DATA/disk1/Datasets/VOCdevkit/'

# dataset settings
data = dict(
    test=dict(
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'])
)

model = dict(bbox_head=dict(num_classes=20))
# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

checkpoint_config = dict(interval=1)

lr_config = dict(policy='step', step=[2])

# runtime settings
epoch_minmax = [3, 1]
evaluation = dict(interval=epoch_minmax[0], metric='mAP')

log_config = dict(interval=50)
l_repeat = 2
u_repeat = 2

# my config
work_dir = './work_dirs/retina'
active_learning = dict(
    cycles=[0, 1, 2, 3, 4, 5, 6],
    budget=16551//40,
    init_num=16551//20,
    epoch=2,
)
