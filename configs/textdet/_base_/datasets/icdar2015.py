# icdar2015_textdet_data_root = 'data/icdar2015'
#
#
# icdar2015_textdet_train = dict(
#     type='OCRDataset',
#     data_root=icdar2015_textdet_data_root,
#     ann_file='textdet_train.json',
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=None)
#
# icdar2015_textdet_test = dict(
#     type='OCRDataset',
#     data_root=icdar2015_textdet_data_root,
#     ann_file='textdet_test.json',
#     test_mode=True,
#     pipeline=None)


ic15_det_data_root = 'data/det/mini_icdar2015/imgs'



icdar2015_textdet_train = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,
    ann_file='instances_training.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

icdar2015_textdet_test = dict(
    type='OCRDataset',
    data_root=ic15_det_data_root,
    ann_file='instances_test.json',
    test_mode=True,
    pipeline=None)
