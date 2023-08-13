# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmdetection3d/blob/v1.0.0rc3/mmdet3d/models/losses/uncertain_smooth_l1_loss.py
# Taken from commit: 0287048a375966b321ad4ce6be4f5ea5c3e84212

import torch
from torch import nn as nn

from typing import Optional


def uncertain_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss with uncertainty.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        sigma (torch.Tensor): The sigma for uncertainty.
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), (
        "The size of pred " f"{pred.size()}, target {target.size()}, and sigma {sigma.size()} " "are inconsistent."
    )
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = torch.exp(-sigma) * loss + alpha * sigma

    return loss


def uncertain_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma: torch.Tensor,
    task: str,
    alpha: float = 1.0,
) -> torch.Tensor:
    """L1 loss with uncertainty.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        sigma (torch.Tensor): The sigma for uncertainty.
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert target.numel() > 0
    assert pred.size() == target.size() == sigma.size(), (
        "The size of pred " f"{pred.size()}, target {target.size()}, and sigma {sigma.size()} " "are inconsistent."
    )
    # sigma 0~1clip
    if task == "wlh":
        sigma_clip = torch.clamp(sigma, -1, 1)
        # sigma_clip = torch.zeros_like(sigma_clip)
        loss = torch.abs(pred - target)
        # loss = torch.exp(-sigma_clip) * loss + alpha * sigma_clip
    elif task == "rp":
        sigma_clip = torch.clamp(sigma, -20, 20)
        # sigma_clip = torch.zeros_like(sigma_clip)
        loss = torch.abs(pred - target)
        # loss = torch.exp(-sigma_clip) * loss + 1.0 * sigma_clip
        loss = loss
    else:
        # sigma_clip = torch.clamp(sigma, -40, 40)
        sigma_clip = sigma
        loss = torch.abs(pred - target)
        loss = torch.exp(-sigma_clip) * loss + alpha * sigma_clip
    return loss


class UncertainSmoothL1Loss(nn.Module):
    """Smooth L1 loss with uncertainty.
    Please refer to `PGD <https://arxiv.org/abs/2107.14160>`_ and
    `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics <https://arxiv.org/abs/1705.07115>`_ for more details.
    Args:
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super(UncertainSmoothL1Loss, self).__init__()
        assert reduction in ["none", "sum", "mean"]
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            sigma (torch.Tensor): The sigma for uncertainty.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        loss_bbox = self.loss_weight * uncertain_smooth_l1_loss(
            pred,
            target,
            weight,
            sigma=sigma,
            alpha=self.alpha,
            beta=self.beta,
        )
        return loss_bbox


class UncertainL1Loss(nn.Module):
    """L1 loss with uncertainty.
    Args:
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.
    """

    def __init__(self, alpha=1.0, reduction="mean", loss_weight=1.0):
        super(UncertainL1Loss, self).__init__()
        assert reduction in ["none", "sum", "mean"]
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        task,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            sigma (torch.Tensor): The sigma for uncertainty.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, "none", "mean", "sum")
        loss_bbox = self.loss_weight * uncertain_l1_loss(pred, target, sigma, task, alpha=self.alpha)
        return loss_bbox
