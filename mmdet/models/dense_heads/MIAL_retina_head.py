import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from ..builder import HEADS
from .anchor_head import AnchorHead
import torch
import torch.nn.functional as F
from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from ..losses import smooth_l1_loss


@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs1 = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mil_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs1.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.cls_convs2.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.mil_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls1 = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_cls2 = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.retina_mil_c = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        # self.retina_mil_l = nn.Conv2d(
        #     self.feat_channels,
        #     self.num_anchors * self.cls_out_channels,
        #     3,
        #     padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs1:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs2:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls1, std=0.01, bias=bias_cls)
        normal_init(self.retina_cls2, std=0.01, bias=bias_cls)
        # normal_init(self.retina_mil_l, std=0.01, bias=bias_cls)
        normal_init(self.retina_mil_c, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat1 = x
        cls_feat2 = x
        reg_feat = x
        mil_feat = x
        for cls_conv1 in self.cls_convs1:
            cls_feat1 = cls_conv1(cls_feat1)
        for cls_conv2 in self.cls_convs2:
            cls_feat2 = cls_conv2(cls_feat2)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        for mil_conv in self.mil_convs:
            mil_feat = mil_conv(mil_feat)
        cls_score1 = self.retina_cls1(cls_feat1)
        cls_score2 = self.retina_cls2(cls_feat2)
        bbox_pred = self.retina_reg(reg_feat)

        nImg = cls_score1.shape[0]
        mil_score_c = self.retina_mil_c(mil_feat)
        # mil_score_l = self.retina_mil_l(mil_feat)
        mil_score_l = (cls_score1 + cls_score2) / 2
        mil_score_l = mil_score_l.detach()
        mil_score_c = mil_score_c.permute(0, 2, 3,
                                          1).reshape(nImg, -1, self.cls_out_channels)
        mil_score_l = mil_score_l.permute(0, 2, 3,
                                          1).reshape(nImg, -1, self.cls_out_channels)
        mil_score = mil_score_c.softmax(2) * mil_score_l.sigmoid().max(2, keepdim=True)[0].softmax(1)

        return cls_score1, cls_score2, bbox_pred, mil_score
