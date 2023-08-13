# Copyright (c) OpenMMLab. All rights reserved.
from loguru import logger

import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F


def multibin_loss(pred_diff, pred_conf, gt_diff, gt_conf, num_bins=8, dtype=None, multihead=False):
    """_summary_

    Args:
        pred_diff (_type_): [N, num_bins, 2(cos,sin)]
        pred_conf (_type_): [N, num_bins]
        gt_diff (_type_): [N, num_bins, 2(cos,sin)]
        gt_conf (_type_): [N, num_bins]
        num_bins (int, optional): number of bins. Defaults to 8.

    Returns:
        _type_: classify_loss, regression_loss
    """
    cls_ce_loss = 0
    reg_losses = 0
    reg_cnt = 0
    if gt_diff.shape[0] > 0:
        num_object = gt_diff.shape[0]
    else:
        num_object = 1
    use_softlaebl = False
    gt_flag = torch.where(gt_conf == 1)
    epsilon = 0.05
    if use_softlaebl:
        norm = norm_dist(num_bins, gt_flag[1], sigma=1.0)
        soft_label = norm  # (1-epsilon)*gt_conf + epsilon*norm
        cls_ce_loss = -torch.mean(soft_label * torch.log(F.softmax(pred_conf + 1e-6, dim=1)))
    else:
        # conf loss
        cls_ce_loss = F.cross_entropy(pred_conf, gt_conf, reduction="mean")
        # cls_ce_loss = -torch.sum(gt_conf*torch.log(F.softmax(pred_conf+1e-6,dim=1)))
    # regression loss
    # cls_ce_loss = F.cross_entropy(
    #    pred_conf,
    #    soft_label,
    #    reduction='mean')
    # L1
    reg_losses = F.l1_loss(pred_diff, gt_diff, reduce=False, size_average=False) * F.softmax(pred_conf).unsqueeze(2)

    # reg_losses = F.l1_loss(pred_diff*gt_conf.unsqueeze(2),gt_diff*gt_conf.unsqueeze(2),reduce=False,size_average=False)
    # alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
    """for (obj_idx,bin_idx) in zip(gt_flag[0],gt_flag[1]):
        if multihead:
            logger.info(pred_diff.shape)
            logger.info(torch.abs(pred_diff[obj_idx, :, 0]-gt_diff[obj_idx, :, 0]).shape)
            logger.info(pred_conf.shape)
            reg_loss = \
                torch.abs(pred_diff[obj_idx, :, 0]-gt_diff[obj_idx, :, 0])*pred_conf[obj_idx] + \
                torch.abs(pred_diff[obj_idx, :, 1]-gt_diff[obj_idx, :, 1])*pred_conf
            reg_loss = reg_loss.mean()
        elif use_l1:
            reg_loss = \
                F.l1_loss(pred_diff[obj_idx, 0], gt_diff[obj_idx, bin_idx, 0]) + \
                F.l1_loss(pred_diff[obj_idx, 1], gt_diff[obj_idx, bin_idx, 1])
        elif use_cos:
            target_diff_theta = torch.atan2(gt_diff[obj_idx, bin_idx, 1], gt_diff[obj_idx, bin_idx, 0])
            pred_diff_theta = torch.atan2(pred_diff[obj_idx, 1], pred_diff[obj_idx, 0])
            reg_loss = \
                - torch.cos(target_diff_theta - pred_diff_theta)
        reg_losses += reg_loss"""
    reg_losses = reg_losses.mean()
    loss = cls_ce_loss / num_bins + reg_losses / num_object
    return cls_ce_loss / num_object, reg_losses / num_object


def norm_dist(num_bin, mu, sigma):
    """_summary_

    Args:
        num_bin (int):
        mu (class_label): [N]
        sigma (flaot):

    Returns:
        _type_: [N, num_bins]
    """
    x = torch.arange(num_bin).type_as(mu).unsqueeze(0).repeat(mu.shape[0], 1)
    mu = mu.unsqueeze(1)
    return (
        torch.exp(-((x - mu) ** 2)) / (2 * torch.pi * sigma**2)
        + torch.exp(-((x - mu - num_bin) ** 2)) / (2 * torch.pi * sigma**2)
        + torch.exp(-((x - mu + num_bin) ** 2)) / (2 * torch.pi * sigma**2)
    )


class MultiBinLoss(nn.Module):
    """Multi-Bin Loss for orientation.
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'none'.
        loss_weight (float, optional): The weight of loss. Defaults
            to 1.0.
    """

    def __init__(self, num_bin=8, overlap=5, loss_weight=1.0, multihead=False):
        super(MultiBinLoss, self).__init__()
        self.num_bin = num_bin
        self.overlap = overlap / 180 * torch.pi
        self.loss_weight = loss_weight
        self.multihead = multihead

    def yaw_preprocess(self, target, dtype):
        """_summary_

        Args:
            target (_type_): shape [B,N,2]

        Returns:
            _type_: _description_
        """
        num_objects = target.shape[0]
        target_theta = torch.atan2(target[:, 1], target[:, 0])
        target_theta = torch.stack([theta if theta >= 0 else theta + 2 * torch.pi for theta in target_theta])
        # logger.info(target_theta*180/np.pi)
        bin_ranges, bins_center = self.make_bins(dtype)
        bin_idxs = []
        for i in range(num_objects):
            bin_idxs.append(self.make_bin_idx(bin_ranges, target_theta[i]))
        multibin_conf = torch.zeros((num_objects, self.num_bin)).type(dtype)
        multibin_diff = torch.zeros((num_objects, self.num_bin, 2)).type(dtype)
        for i, bin_idx in enumerate(bin_idxs):
            diff_theta = target_theta[i] - bins_center
            multibin_conf[i][bin_idx] = 1
            multibin_diff[i, :, :] = torch.stack([torch.cos(diff_theta), torch.sin(diff_theta)]).T.view(-1, 2)
        return multibin_conf, multibin_diff

    def make_bins(self, dtype):
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        bins_center = torch.zeros(self.num_bin).type(dtype)
        interval = 2 * torch.pi / self.num_bin
        for i in range(1, self.num_bin):
            bins_center[i] = i * interval
        bins_center += interval / 2
        bin_ranges = []
        for i in range(0, self.num_bin):
            bin_ranges.append(
                (
                    (i * interval - self.overlap) % (2 * torch.pi),
                    (i * interval + interval + self.overlap) % (2 * torch.pi),
                )
            )
        return bin_ranges, bins_center

    def make_bin_idx(self, bin_ranges, angle):
        bin_idxs = []
        for bin_idx, bin_range in enumerate(bin_ranges):
            if self.is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)
        return bin_idxs

    def is_between(self, min, max, angle):
        max = (max - min) if (max - min) > 0 else (max - min) + 2 * torch.pi
        angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * torch.pi
        return angle <= max

    def forward(self, pred_diff, pred_conf, target):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            num_dir_bins (int): Number of bins to encode direction angle.
        """
        if target.shape[0] == 0:
            dtype = target.type()
            return torch.tensor(0).type(dtype), torch.tensor(0).type(dtype)
        else:
            dtype = target.type()
            target_conf, target_diff = self.yaw_preprocess(target, dtype)
            loss_conf, loss_rot = multibin_loss(
                pred_diff,
                pred_conf,
                target_diff.type(dtype),
                target_conf.type(dtype),
                num_bins=self.num_bin,
                dtype=dtype,
                multihead=self.multihead,
            )
            loss = loss_conf + self.loss_weight * loss_rot
            return loss_rot, loss_conf
