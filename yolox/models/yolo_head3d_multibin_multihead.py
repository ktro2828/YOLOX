#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.losses import IOUloss, MultiBinLoss

from typing import List, Optional

from .yolo_head3d import YOLOXHead3d


class YOLOXHead3dMultiBinMultiHead(YOLOXHead3d):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        ph="3d",
        num_bin=8,
        overlap=0,
        diff_weight=1,
        depthwise=False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=strides,
            in_channels=in_channels,
            act=act,
            ph=ph,
            depthwise=depthwise,
        )

        # --- multibin loss --- #
        self.num_bin = num_bin
        self.overlap = overlap
        self.diff_weight = diff_weight
        # --------------------- #

        # ------------- loss function ---------------- #
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.multibin_loss = MultiBinLoss(
            num_bin=self.num_bin, overlap=self.overlap, loss_weight=self.diff_weight, multihead=True
        )
        self.softmax = nn.Softmax(dim=1)
        self.epoch = 0

    def forward(
        self,
        xin: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        imgs: Optional[torch.Tensor] = None,
    ):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        self.epoch += 1

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.td_cls_convs, self.td_reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            # --- 3d prediction --- #
            ddd_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.td_cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.td_reg_preds[k](reg_feat)
            obj_output = self.td_obj_preds[k](reg_feat)

            # --- 3d prediction --- #
            ddd_feat = self.ddd_convs[k](ddd_x)
            yaw_conf_output = self.ddd_yaw_conf_preds[k](ddd_feat)
            yaw_output = self.ddd_yaw_preds[k](ddd_feat)
            rp_output = self.ddd_rp_preds[k](ddd_feat)
            wlh_output = self.ddd_wlh_preds[k](ddd_feat)

            if self.training:
                # output shape [N,num_anchor,4+1+8+3+4+2+8=30,H,W]
                # n_ch = 4(xyhw)+1(obj)+8(class)+3(wlh)+4(rp)+2(cos,sin)+8(num_bin)=30
                output = torch.cat(
                    [reg_output, obj_output, cls_output, wlh_output, rp_output, yaw_output, yaw_conf_output], 1
                )
                output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                # ddd output
                output = torch.cat(
                    [
                        reg_output,
                        obj_output.sigmoid(),
                        cls_output.sigmoid(),
                        wlh_output,
                        rp_output,
                        yaw_output,
                        yaw_conf_output,
                    ],
                    1,
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        # n_ch = 5 + self.num_classes
        # --- 3d --- #
        # n_ch = 4(xyhw)+1(obj)+8(class)+3(wlh)+4(rp)+2(cos,sin)+8(num_bin)=30
        n_ch = 12 + self.num_bin * 2 + self.num_classes + self.num_bin
        # n_ch = 14 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        # xyhw をグリッドから座標に戻している
        output[..., :2] = (output[..., :2] + grid) * stride
        # hw
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        # --- ddd --- #
        # rp(top_center)
        output[..., 5 + 3 + self.num_classes : 5 + 3 + 2 + self.num_classes] = (
            output[..., 5 + 3 + self.num_classes : 5 + 3 + 2 + self.num_classes] + grid
        ) * stride
        # rp(bottom_center)
        output[..., 5 + 3 + 2 + self.num_classes : 5 + 3 + 4 + self.num_classes] = (
            output[..., 5 + 3 + 2 + self.num_classes : 5 + 3 + 4 + self.num_classes] + grid
        ) * stride
        return output, grid

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # --- 2d --- #
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5 : 5 + self.num_classes]  # [batch, n_anchors_all, n_cls]

        # --- 3d prediction --- #
        wlh_preds = outputs[:, :, 5 + self.num_classes : 5 + 3 + self.num_classes]  # [batch, n_anchors_all, 3]
        rp_preds = outputs[:, :, 5 + 3 + self.num_classes : 5 + 3 + 4 + self.num_classes]  # [batch, n_anchors_all, 4]
        yaw_preds = outputs[
            :, :, 5 + 3 + 4 + self.num_classes : 5 + 3 + 4 + 2 * self.num_bin + self.num_classes
        ]  # [batch, n_anchors_all, 2]
        yaw_conf_preds = outputs[
            :, :, 5 + 3 + 4 + 2 * self.num_bin + self.num_classes :
        ]  # [batch, n_anchors_all, num_bin]
        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        # ----3d ----
        wlh_targets = []
        rp_targets = []
        yaw_targets = []
        theta_ray_targets = []

        num_fg = 0.0
        num_gts = 0.0
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                # --- 3d prediction --- #
                wlh_target = outputs.new_zeros((0, 3))
                rp_target = outputs.new_zeros((0, 4))
                yaw_target = outputs.new_zeros((0, 2))
                theta_ray_target = outputs.new_zeros((0, 2))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                # --- 3d prediction --- #
                if self.lr_ph == "3d":
                    gt_wlh_per_image = labels[batch_idx, :num_gt, 5:8]
                    gt_rp_per_image = labels[batch_idx, :num_gt, 8:12]
                    gt_yaw_per_image = labels[batch_idx, :num_gt, 12:14]
                    gt_theta_ray_per_image = labels[batch_idx, :num_gt, 14:16]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        # --- 3d --- #
                        wlh_preds,
                        rp_preds,
                        yaw_preds,
                        yaw_conf_preds,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        # --- 3d --- #
                        wlh_preds,
                        rp_preds,
                        yaw_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                # --- 3d TODO --- #
                if self.lr_ph == "3d":
                    wlh_target = gt_wlh_per_image[matched_gt_inds]
                    rp_target = gt_rp_per_image[matched_gt_inds]
                    yaw_target = gt_yaw_per_image[matched_gt_inds]
                    theta_ray_target = gt_theta_ray_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            if self.lr_ph == "3d":
                wlh_targets.append(wlh_target)
                rp_targets.append(rp_target)
                yaw_targets.append(yaw_target)
                theta_ray_targets.append(theta_ray_target)

            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        if self.lr_ph == "3d":
            wlh_targets = torch.cat(wlh_targets, 0)
            rp_targets = torch.cat(rp_targets, 0)
            yaw_targets = torch.cat(yaw_targets, 0)
            theta_ray_targets = torch.cat(theta_ray_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fg
        # --- 3d prediction --- # TODO
        if self.lr_ph == "3d":
            loss_wlh = (self.l1_loss(wlh_preds.view(-1, 3)[fg_masks], wlh_targets)).sum() / num_fg
            loss_rp = (self.l1_loss(rp_preds.view(-1, 4)[fg_masks], rp_targets)).sum() / num_fg
            loss_yaw, loss_yaw_conf = self.multibin_loss(
                yaw_preds.view(-1, self.num_bin, 2)[fg_masks],
                yaw_conf_preds.view(-1, self.num_bin)[fg_masks],
                yaw_targets,
            )
            loss_yaw = loss_yaw.sum() / num_fg
            loss_yaw_conf = loss_yaw_conf.sum() / num_fg

        if self.use_l1:
            loss_l1 = (self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        # epochs = np.arange(1, 101)
        # alpha = 0.02
        # beta = 0.6
        # weight_offset = np.hstack(
        #     [alpha * epochs[epochs * alpha < beta], beta * np.ones(len(alpha * epochs[epochs * alpha >= beta]))]
        # )
        # tmp_epoch = self.epoch if self.epoch < 100 else 99
        reg_weight = 5.0
        rp_weight = 1
        yaw_weight = 150000  # *(weight_offset[tmp_epoch])
        wlh_weight = 12
        yaw_conf_weight = 30000  # *(1-weight_offset[tmp_epoch])
        loss_ddd_obj = 0
        if self.lr_ph == "2d":
            loss = reg_weight * loss_iou + loss_obj + loss_cls
            loss_wlh = 0
            loss_rp = 0
            loss_yaw = 0
            loss_yaw_conf = 0
        else:
            loss = wlh_weight * loss_wlh + rp_weight * loss_rp + yaw_weight * loss_yaw + yaw_conf_weight * loss_yaw_conf
            loss_iou = 0
            loss_obj = 0
            loss_cls = 0

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            wlh_weight * loss_wlh,
            rp_weight * loss_rp,
            yaw_weight * loss_yaw,
            yaw_conf_weight * loss_yaw_conf,
            loss_ddd_obj,
            num_fg / max(num_gts, 1),
        )
