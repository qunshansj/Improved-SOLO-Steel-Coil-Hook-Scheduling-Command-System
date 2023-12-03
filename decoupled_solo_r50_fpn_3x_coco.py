python
class SoloModel:
    def __init__(self):
        self.num_classes = 80
        self.in_channels = 256
        self.stacked_convs = 7
        self.feat_channels = 256
        self.strides = [8, 8, 16, 32, 32]
        self.scale_ranges = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
        self.pos_scale = 0.2
        self.num_grids = [40, 36, 24, 16, 12]
        self.cls_down_index = 0
        self.loss_mask = dict(
            type='DiceLoss', use_sigmoid=True, activate=False, loss_weight=3.0)
        self.loss_cls = dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
