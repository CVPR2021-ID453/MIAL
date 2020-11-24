from abc import ABCMeta, abstractmethod

import torch.nn as nn
from tools.utils import losstype


class BaseDenseHead(nn.Module, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    @abstractmethod
    def get_bboxes(self, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs1, outs2, reg, mil_score = self(x)
        # init loss two inputs
        if losstype.losstype == 0:
            outs = (outs1, reg, mil_score)
            if gt_labels is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            losses1 = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            outs = (outs2, reg, mil_score)
            if gt_labels is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            losses2 = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_cls = list(map(lambda m, n: (m + n)/2, losses1['loss_cls'], losses2['loss_cls']))
            losses_bbox = list(map(lambda m, n: (m + n)/2, losses1['loss_bbox'], losses2['loss_bbox']))
            losses_mil = list(map(lambda m, n: (m + n)/2, losses1['loss_mil'], losses2['loss_mil']))
            losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_mil=losses_mil)

        # agreement three inputs
        elif losstype.losstype == 1:
            outs = ((outs1, outs2), reg, mil_score)
            if gt_labels is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss = self.loss_min(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses = dict(loss_cls=loss['loss_cls'],
                          loss_bbox=loss['loss_bbox'],
                          loss_agr=loss['loss_agr'],
                          loss_mil=loss['loss_mil'])

        # discrepancy
        else:
            outs = ((outs1, outs2), reg, mil_score)
            if gt_labels is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss = self.loss_max(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses = dict(loss_cls=loss['loss_cls'], loss_bbox=loss['loss_bbox'], loss_dsc=loss['loss_dsc'])

        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
