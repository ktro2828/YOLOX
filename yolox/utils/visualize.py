#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

import torch

from yolox.data.datasets.coco import Box, view_points

from pyquaternion import Quaternion

__all__ = ["vis", "vis2", "ddd_vis"]


def ddd_vis(
    img,
    wlhs,
    rps,
    yaws,
    clss,
    scores,
    cam_intrinsic,
    anns,
    image_name,
    use_8rp=False,
    dd_boxes=None,
    use_original_nms=True,
    use_multibin=False,
):
    """visualization for 3d detection

    Args:
        img (np.array): img array [H,W,3]
        wlhs (list): width, length, height [num_object,3]
        rps (list): reference point [num_object, 4]
        yaws (list): yaw (cos,sin) [num_object, 2]
        clss (list): class [num_object, 1]
        cam_intrinsic (np.array): camera intrinsic
        anns[list] : nuscenes like annotations
    """
    bev_img = np.zeros_like(img)
    bev_img = ann_bev(bev_img, anns, image_name)
    # plot_ann_rp(img,anns,cam_intrinsic,image_name)
    f = cam_intrinsic[0][0]
    cx = cam_intrinsic[0][2]
    # cy = cam_intrinsic[1][2]
    width, _ = img.shape[1], img.shape[0]
    fov = 2 * np.arctan2(width / 2, f)
    boxes = []
    yaws_l = []
    clss_l = []
    score_l = []
    dd_boxes_l = []
    for i in range(len(wlhs)):
        wlh = wlhs[i]
        rp = rps[i]
        plot_rp(img, rp, color=(0, 0, 255))
        yaw = yaws[i]
        cls = clss[i]
        score = scores[i]
        dd_box = dd_boxes[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for cnt, (t4_label, cls_id) in enumerate(t4label_to_index.items()):
            cv2.rectangle(
                img,
                (25, 10 * (2 * cnt + 1)),
                (50, 10 * (2 * cnt + 1) + 10),
                (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
                -1,
            )
            cv2.putText(
                img,
                t4_label,
                (60, 10 * (2 * cnt + 1) + 8),
                font,
                0.4,
                (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
                thickness=1,
            )
        use_8rp = False
        if use_8rp:
            (
                front_left_top_x,
                front_left_top_y,
                front_right_top_x,
                front_right_top_y,
                front_right_bottom_x,
                front_right_bottom_y,
                front_left_bottom_x,
                front_left_bottom_y,
                back_left_top_x,
                back_left_top_y,
                back_right_top_x,
                back_right_top_y,
                back_right_bottom_x,
                back_right_bottom_y,
                back_left_bottom_x,
                back_left_bottom_y,
            ) = rp
            top_center_x, top_center_y = prog(
                front_left_top_x,
                front_left_top_y,
                front_right_top_x,
                front_right_top_y,
                back_left_top_x,
                back_left_top_y,
                back_right_top_x,
                back_right_top_y,
            )
            bottom_center_x, bottom_center_y = prog(
                front_right_bottom_x,
                front_right_bottom_y,
                front_left_bottom_x,
                front_left_bottom_y,
                back_right_bottom_x,
                back_right_bottom_y,
                back_left_bottom_x,
                back_left_bottom_y,
            )
            bbox_center = [(top_center_x + bottom_center_x) / 2, (top_center_y + bottom_center_y) / 2]
        else:
            top_center_x, top_center_y, bottom_center_x, bottom_center_y = rp
            top_center_x, top_center_y, bottom_center_x, bottom_center_y = (
                int(top_center_x),
                int(top_center_y),
                int(bottom_center_x),
                int(bottom_center_y),
            )
            bbox_center = [(top_center_x + bottom_center_x) / 2, (top_center_y + bottom_center_y) / 2]  # center pixel
        if bbox_center[0] - cx < 0:
            theta_ray_rad = np.arctan(abs(bbox_center[0] - cx) / f) + np.pi / 2
        else:
            theta_ray_rad = np.pi / 2 - np.arctan((bbox_center[0] - cx) / f)
        theta_ray = [np.cos(theta_ray_rad), np.sin(theta_ray_rad)]
        if top_center_x > 0 and top_center_y > 0 and bottom_center_x > 0 and bottom_center_y > 0:
            offset_ori = Quaternion(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
            if use_multibin:
                global_yaw_rad = yaw  # +theta_ray_rad
            else:
                global_yaw_rad = localyaw2globalyaw(yaw, theta_ray)
            # global_yaw_deg = global_yaw_rad * 180 / np.pi
            ori = Quaternion(axis=[0, -1, 0], angle=global_yaw_rad)
            infer_center = inference_center(
                [top_center_x, top_center_y, bottom_center_x, bottom_center_y], wlh, cam_intrinsic
            )  # center(x,y,z) in camera coordinate [m]
            color = _COLORS[int(cls.item())] * 255
            box = Box(infer_center, list(wlh.numpy()), ori * offset_ori)  # orientation := yaw
            boxes.append(box)
            yaws_l.append(global_yaw_rad)
            clss_l.append(cls)
            score_l.append(score)
            dd_boxes_l.append(np.array(dd_box))
            if use_original_nms:
                continue
            else:
                box.render_cv2(
                    img,
                    cam_intrinsic,
                    normalize=True,
                    colors=(
                        color.astype(np.uint8).tolist(),
                        color.astype(np.uint8).tolist(),
                        color.astype(np.uint8).tolist(),
                    ),
                    yaw=round(list(wlh.numpy())[2], 1),
                )
                bev_img = bev(bev_img, box, global_yaw_rad, color)
    if use_original_nms and len(dd_boxes_l) != 0:
        boxes, yaws_l, clss_l, _, _ = nms_original(boxes, yaws_l, clss_l, dd_boxes_l, score_l)
        for box, global_yaw_rad, cls in zip(boxes, yaws_l, clss_l):
            color = _COLORS[int(cls.item())] * 255
            box.render_cv2(
                img,
                cam_intrinsic,
                normalize=True,
                colors=(
                    color.astype(np.uint8).tolist(),
                    color.astype(np.uint8).tolist(),
                    color.astype(np.uint8).tolist(),
                ),
                yaw=round(list(wlh.numpy())[2], 1),
            )
            bev_img = bev(bev_img, box, box.euler[2], color)
    bev_img = bev_postprocess(bev_img, fov)
    return img, bev_img


def localyaw2globalyaw(yaw, theta):
    """function to convert loacal yaw to global yaw
    Args:
        yaw (tensor): nn output to predict local yaw [N,2] N=number of objects
        theta (tensor): theta ray [N,2]
    Returns:
        global yaw (tensor): global yaw [N,2]
    """
    global_yaw = torch.zeros_like(yaw)
    cos_alpha = yaw[0]
    sin_alpha = yaw[1]
    cos_beta = theta[0]
    sin_beta = theta[1]
    global_yaw[0] = cos_alpha * cos_beta - sin_alpha * sin_beta
    global_yaw[1] = sin_alpha * cos_beta + cos_alpha * sin_beta
    global_yaw_rad = np.arctan2(global_yaw[1], global_yaw[0])
    return global_yaw_rad


def inference_center(rp, wlh, cam_intrinsic):
    """inference center of objects [m]

    Args:
        rp (): reference point (top center, bottom_center)
        wlh (_type_): width, length, height [m]
        cam_intrinsic (_type_): camera intrinsic (inner parameter)

    Returns:
        _type_: inference center [m]
    """
    top_center_x, top_center_y, bottom_center_x, bottom_center_y = rp
    w, l, h = wlh
    f = cam_intrinsic[0][0]
    cx = cam_intrinsic[0][2]
    cy = cam_intrinsic[1][2]
    vec_a = np.array([top_center_x - cx, top_center_y - cy, f])
    vec_b = np.array([bottom_center_x - cx, bottom_center_y - cy, f])
    beta = np.arccos(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))
    d = h / (2 * np.tan(beta / 2))  # depth [m]
    bbox_center = [(top_center_x + bottom_center_x) / 2, (top_center_y + bottom_center_y) / 2]
    if bbox_center[0] - cx < 0:
        theta_ray = np.arctan(abs(bbox_center[0] - cx) / f) + np.pi / 2
    else:
        theta_ray = np.pi / 2 - np.arctan((bbox_center[0] - cx) / f)
    if bbox_center[1] - cy < 0:
        theta_ray2 = -np.arctan(abs(bbox_center[1] - cy) / f)
    else:
        theta_ray2 = np.arctan((bbox_center[1] - cy) / f)
    infer_center = [d * np.cos(theta_ray), d * np.tan(theta_ray2), d * np.sin(theta_ray)]  # := [x,y,z]
    return infer_center


def bev(img, box, yaw, color, score=1.0):
    """genarate img to bird's eye view

    Args:
        img (np.array) [H,W,3]
        box (_type_): Box
        yaw (_type) : yaw radiun
    """
    shape = img.shape
    # ハイパラ
    scale = 10
    # score = score.float().item()
    color = color * score
    color = color.astype(np.uint8).tolist()
    w, l, h = box.wlh * scale
    x, y, z = box.center * scale
    x += shape[1] / 2
    bev_center = np.array([x, z])
    corner = np.array([[x - l / 2, z + w / 2], [x + l / 2, z + w / 2], [x - l / 2, z - w / 2], [x + l / 2, z - w / 2]])
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corner[0] = np.dot(rotation_matrix, (corner[0] - bev_center)) + bev_center
    corner[1] = np.dot(rotation_matrix, (corner[1] - bev_center)) + bev_center
    corner[2] = np.dot(rotation_matrix, (corner[2] - bev_center)) + bev_center
    corner[3] = np.dot(rotation_matrix, (corner[3] - bev_center)) + bev_center
    # corner[rear left, front left, rear right, front right]
    cv2.line(img, corner[0].astype("int"), corner[1].astype("int"), color, thickness=2)
    cv2.line(img, corner[0].astype("int"), corner[2].astype("int"), color, thickness=2)
    cv2.line(img, corner[1].astype("int"), corner[3].astype("int"), color, thickness=2)
    cv2.line(img, corner[2].astype("int"), corner[3].astype("int"), color, thickness=2)
    cv2.line(
        img,
        ((corner[1] + corner[3]) / 2).astype("int"),
        ((corner[0] + corner[1] + corner[2] + corner[3]) / 4).astype("int"),
        color,
        thickness=2,
    )
    return img


def ann_bev(img, anns, image_name=None):
    shape = img.shape
    # ハイパラ
    scale = 10
    # for sequence images
    if image_name:
        for i, ann in enumerate(anns):
            if ann["file_name"] in image_name:
                w, l, _ = [ann["bbox_cam3d"][5] * scale, ann["bbox_cam3d"][3] * scale, ann["bbox_cam3d"][4] * scale]
                x, _, z = [ann["bbox_cam3d"][0] * scale, ann["bbox_cam3d"][1] * scale, ann["bbox_cam3d"][2] * scale]
                yaw = -ann["bbox_cam3d"][-1]
                x += shape[1] / 2
                bev_center = np.array([x, z])
                corner = np.array([[-l / 2, w / 2], [l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
                rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                corner[0] = np.dot(rotation_matrix, (corner[0])) + bev_center
                corner[1] = np.dot(rotation_matrix, (corner[1])) + bev_center
                corner[2] = np.dot(rotation_matrix, (corner[2])) + bev_center
                corner[3] = np.dot(rotation_matrix, (corner[3])) + bev_center
                # corner[rear left, front left, rear right, front right]
                cv2.line(img, corner[0].astype("int"), corner[1].astype("int"), (0, 0, 255), thickness=2)
                cv2.line(img, corner[0].astype("int"), corner[2].astype("int"), (0, 0, 255), thickness=2)
                cv2.line(img, corner[1].astype("int"), corner[3].astype("int"), (0, 0, 255), thickness=2)
                cv2.line(img, corner[2].astype("int"), corner[3].astype("int"), (0, 0, 255), thickness=2)
                cv2.line(
                    img,
                    ((corner[1] + corner[3]) / 2).astype("int"),
                    ((corner[0] + corner[1] + corner[2] + corner[3]) / 4).astype("int"),
                    (0, 0, 255),
                    thickness=2,
                )
    else:  # for one image
        for ann in anns:
            w, l, _ = [ann["bbox_cam3d"][5] * scale, ann["bbox_cam3d"][3] * scale, ann["bbox_cam3d"][4] * scale]
            x, _, z = [ann["bbox_cam3d"][0] * scale, ann["bbox_cam3d"][1] * scale, ann["bbox_cam3d"][2] * scale]
            yaw = -ann["bbox_cam3d"][-1]
            x += shape[1] / 2
            bev_center = np.array([x, z])
            corner = np.array(
                [[x - l / 2, z + w / 2], [x + l / 2, z + w / 2], [x - l / 2, z - w / 2], [x + l / 2, z - w / 2]]
            )
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            corner[0] = np.dot(rotation_matrix, (corner[0] - bev_center)) + bev_center
            corner[1] = np.dot(rotation_matrix, (corner[1] - bev_center)) + bev_center
            corner[2] = np.dot(rotation_matrix, (corner[2] - bev_center)) + bev_center
            corner[3] = np.dot(rotation_matrix, (corner[3] - bev_center)) + bev_center
            # corner[rear left, front left, rear right, front right]
            cv2.line(img, corner[0].astype("int"), corner[1].astype("int"), (0, 0, 255), thickness=2)
            cv2.line(img, corner[0].astype("int"), corner[2].astype("int"), (0, 0, 255), thickness=2)
            cv2.line(img, corner[1].astype("int"), corner[3].astype("int"), (0, 0, 255), thickness=2)
            cv2.line(img, corner[2].astype("int"), corner[3].astype("int"), (0, 0, 255), thickness=2)
            cv2.line(
                img,
                ((corner[1] + corner[3]) / 2).astype("int"),
                ((corner[0] + corner[1] + corner[2] + corner[3]) / 4).astype("int"),
                (0, 0, 255),
                thickness=2,
            )
    return img


def bev_postprocess(bev_img, fov):
    bev_img = cv2.flip(bev_img, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cnt, (t4_label, cls_id) in enumerate(t4label_to_index.items()):
        cv2.rectangle(
            bev_img,
            (25, 10 * (2 * cnt + 1)),
            (50, 10 * (2 * cnt + 1) + 10),
            (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
            -1,
        )
        cv2.putText(
            bev_img,
            t4_label,
            (60, 10 * (2 * cnt + 1) + 8),
            font,
            0.4,
            (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
            thickness=1,
        )
    cnt += 1
    cv2.rectangle(bev_img, (25, 10 * (2 * cnt + 1)), (50, 10 * (2 * cnt + 1) + 10), (0, 0, 255), -1)
    cv2.drawMarker(
        bev_img,
        position=(int(bev_img.shape[1] / 2), bev_img.shape[0]),
        color=(255, 255, 255),
        markerType=cv2.MARKER_TRIANGLE_DOWN,
        markerSize=20,
        thickness=2,
        line_type=cv2.LINE_4,
    )
    fov_point1 = (0, bev_img.shape[0] - int(bev_img.shape[1] / (2 * np.tan(fov / 2))))
    fov_point2 = (bev_img.shape[1], bev_img.shape[0] - int(bev_img.shape[1] / (2 * np.tan(fov / 2))))
    cv2.line(bev_img, fov_point1, (int(bev_img.shape[1] / 2), bev_img.shape[0]), (255, 255, 255), thickness=2)
    cv2.line(bev_img, fov_point2, (int(bev_img.shape[1] / 2), bev_img.shape[0]), (255, 255, 255), thickness=2)
    cv2.putText(bev_img, "GT", (60, 10 * (2 * cnt + 1) + 8), font, 0.4, (0, 0, 255), thickness=1)
    return bev_img


def plot_rp(img, rp, color):
    cv2.drawMarker(img, (int(rp[0]), int(rp[1])), color=color, markerType=cv2.MARKER_STAR)
    cv2.drawMarker(img, (int(rp[2]), int(rp[3])), color=color, markerType=cv2.MARKER_STAR)


def plot_ann_rp(img, anns, cam_intrinsic, image_name):
    scale = 1
    if image_name:
        for i, ann in enumerate(anns):
            if ann["file_name"] in image_name:
                w, l, h = [ann["bbox_cam3d"][5] * scale, ann["bbox_cam3d"][3] * scale, ann["bbox_cam3d"][4] * scale]
                x, y, z = [ann["bbox_cam3d"][0] * scale, ann["bbox_cam3d"][1] * scale, ann["bbox_cam3d"][2] * scale]
                yaw = -ann["bbox_cam3d"][-1]
                offset_ori = Quaternion(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
                ori = Quaternion(axis=[0, -1, 0], angle=yaw)
                ann_box = Box([x, y, z], [w, l, h], ori * offset_ori)
                rp, _ = calc_2rp(ann_box, cam_intrinsic)
                plot_rp(img, rp, color=(0, 0, 255))
                ann_box.render_cv2(
                    img, cam_intrinsic, normalize=True, colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)), yaw=round(h, 1)
                )
    else:
        for ann in anns:
            w, l, h = [ann["bbox_cam3d"][5] * scale, ann["bbox_cam3d"][3] * scale, ann["bbox_cam3d"][4] * scale]
            x, y, z = [ann["bbox_cam3d"][0] * scale, ann["bbox_cam3d"][1] * scale, ann["bbox_cam3d"][2] * scale]
            yaw = -ann["bbox_cam3d"][-1]
            offset_ori = Quaternion(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
            ori = Quaternion(axis=[0, -1, 0], angle=yaw)
            ann_box = Box([x, y, z], [w, l, h], ori * offset_ori)
            rp, _ = calc_2rp(ann_box, cam_intrinsic)
            plot_rp(img, rp, color=(0, 0, 255))
            ann_box.render_cv2(
                img, cam_intrinsic, normalize=True, colors=((0, 0, 255), (0, 0, 255), (0, 0, 255)), yaw=round(h, 1)
            )


def calc_2rp(box, cam_intrinsic):
    """calclate 2 reference point (top center and bottom center) function

    Args:
        box (Box): box
        cam_intrinsic (_type_): camera intrinsic

    Return:
        [tcx,tcy,bcx,bcy], theta_ray
        (top center x, top center y, bottom center x, bottom center y) pixel
    """
    corners = view_points(box.corners(), cam_intrinsic, normalize=True)[:2, :]
    f = cam_intrinsic[0][0]
    cx = cam_intrinsic[0][2]
    # cy = cam_intrinsic[1][2]
    top_center_x, top_center_y = prog(
        corners[0][0],
        corners[1][0],
        corners[0][5],
        corners[1][5],
        corners[0][1],
        corners[1][1],
        corners[0][4],
        corners[1][4],
    )
    bottom_center_x, bottom_center_y = prog(
        corners[0][2],
        corners[1][2],
        corners[0][7],
        corners[1][7],
        corners[0][3],
        corners[1][3],
        corners[0][6],
        corners[1][6],
    )
    bbox_center = [(top_center_x + bottom_center_x) / 2, (top_center_y + bottom_center_y) / 2]
    if bbox_center[0] - cx < 0:
        theta_ray = np.arctan(abs(bbox_center[0] - cx) / f) + np.pi / 2
    else:
        theta_ray = np.pi / 2 - np.arctan((bbox_center[0] - cx) / f)
    return [top_center_x, top_center_y, bottom_center_x, bottom_center_y], theta_ray


def prog(x1, y1, x2, y2, x3, y3, x4, y4):
    if x1 == x2 and x3 == x4:
        x = y = np.nan
    elif x1 == x2:
        x = x1
        y = (y4 - y3) / (x4 - x3) * (x1 - x3) + y3
    elif x3 == x4:
        x = x3
        y = (y2 - y1) / (x2 - x1) * (x3 - x1) + y1
    else:
        a1 = (y2 - y1) / (x2 - x1)
        a3 = (y4 - y3) / (x4 - x3)
        if a1 == a3:
            x = y = np.nan
        else:
            x = (a1 * x1 - y1 - a3 * x3 + y3) / (a1 - a3)
            y = (y2 - y1) / (x2 - x1) * (x - x1) + y1
    return (x, y)


def giou(a, b):
    # a, bは矩形を表すリストで、a=[xmin, ymin, xmax, ymax]
    ax_mn, ay_mn, ax_mx, ay_mx = a[3][0], a[1][1], a[0][0], a[2][1]
    bx_mn, by_mn, bx_mx, by_mx = b[3][0], b[1][1], b[0][0], b[2][1]
    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w * h

    # unionの面積を計算
    union = a_area + b_area - intersect
    # IoUを計算
    iou = intersect / union

    # convex shape Cの面積を計算
    abx_mn = min(ax_mn, bx_mn)
    aby_mn = min(ay_mn, by_mn)
    abx_mx = max(ax_mx, bx_mx)
    aby_mx = max(ay_mx, by_mx)
    c_area = (abx_mx - abx_mn + 1) * (aby_mx - aby_mn + 1)

    # GIoUを計算
    giou = iou - (c_area - union) / c_area

    return giou


def iou_np(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])

    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:, 0])  # xmin
    aby_mn = np.maximum(a[1], b[:, 1])  # ymin
    abx_mx = np.minimum(a[2], b[:, 2])  # xmax
    aby_mx = np.minimum(a[3], b[:, 3])  # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w * h

    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)
    return iou


def nms_original(bboxes, yaws, clss, dd_boxes, scores):
    # scale = 1
    # corners = []
    dd_boxes = np.array(dd_boxes)
    # print(dd_boxes)
    # start_x = dd_boxes[:, 0]
    # start_y = dd_boxes[:, 1]
    # end_x = dd_boxes[:, 2]
    # end_y = dd_boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    # picked_clss = []
    indexes = []

    # Compute areas of bounding boxes
    # areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    # print("total num",len(bboxes))

    # scoreの昇順(小さい順)の矩形インデックスのリストを取得
    sort_index = np.argsort(score)
    iou_threshold = 0.8
    i = -1  # 未処理の矩形のindex
    while len(sort_index) > 0:
        # print(dd_boxes[sort_index])
        # score最大のindexを取得
        max_scr_ind = sort_index[i]
        # score最大以外のindexを取得
        ind_list = sort_index
        # score最大の矩形それ以外の矩形のIoUを計算
        iou = iou_np(dd_boxes[max_scr_ind], dd_boxes[ind_list])
        # print(iou)

        # IoUが閾値iou_threshold以上の矩形を計算
        del_index = np.where(iou >= iou_threshold)
        indexes.append(sort_index[del_index])
        # IoUが閾値iou_threshold以上の矩形を削除
        sort_index = np.delete(sort_index, del_index)
        if len(list(sort_index)) == 1:
            indexes.append(sort_index)
            break

        # i -= 1 # 未処理の矩形のindexを1減らす

    # return picked_boxes, picked_score
    final_box = []
    final_yaws = []
    final_clss = []

    for index in indexes:  # [object_num, bbox_candidate]
        bev_centers = []
        bev_dims = []
        bev_yaws = []
        score_list = []
        if len(index) == 1:
            best_center = bboxes[index[0]].center
            best_dim = bboxes[index[0]].wlh
            best_yaw = bboxes[index[0]].euler[2]
        else:
            for i in index:
                w, l, h = bboxes[i].wlh
                x, y, z = bboxes[i].center
                roll, pitch, yaw = bboxes[i].euler
                bev_center = np.array([x, z])
                bev_centers.append(np.array([x, y, z]))
                other_index = [j for j in index if j != i]
                other_yaws = np.array([bboxes[o].euler for o in other_index])
                other_centers = np.array([np.array([bboxes[o].center[0], bboxes[o].center[2]]) for o in other_index])
                bev_score = np.sum(np.cos(yaw - other_yaws)) / np.sum(np.linalg.norm(bev_center - other_centers, ord=2))
                bev_dims.append(np.array([w, l, h]))
                bev_yaws.append(yaw)
                score_list.append(bev_score)
            # bev_center_np = np.array(bev_centers)
            # ave_center = np.mean(bev_center_np, axis=0)
            # center_distance = np.linalg.norm(bev_center_np - ave_center, ord=2)
            # center_min_ind = np.argmax(center_distance)
            bev_best_score_index = np.argmin(np.array(score_list))
            best_center = bev_centers[bev_best_score_index]
            best_yaw = bev_yaws[bev_best_score_index]
            best_dim = bev_dims[bev_best_score_index]
            # bev_dim_np = np.array(bev_dims)
            # ave_dim = np.mean(bev_dim_np, axis=0)
            # bev_yaw_np = np.array(bev_yaws)
            # ave_yaw = np.mean(bev_yaw_np, axis=0)

        cls = clss[i]
        ori = Quaternion(axis=[0, -1, 0], angle=best_yaw)
        offset_ori = Quaternion(np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
        box = Box(best_center, best_dim, ori * offset_ori)
        final_box.append(box)
        final_yaws.append(best_yaw)
        final_clss.append(cls)
    return final_box, final_yaws, final_clss, picked_boxes, picked_score


def polygon_area(P):
    """calculate polygon area

    Args:
        P (list): the points of suquare [4,2] (xi,yi) i=0~3

    Returns:
        area (float): the area of polygon
    """
    return abs(sum(P[i][0] * P[i - 1][1] - P[i][1] * P[i - 1][0] for i in range(4))) / 2.0


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


nu_dict = {
    "vehicle.car": "CAR",
    "human.pedestrian.adult": "PEDESTRIAN",
    "movable_object.barrier": "UNKNOWN",
    "movable_object.traffic_cone": "UNKNOWN",
    "vehicle.truck": "TRUCK",
    "vehicle.bicycle": "BICYCLE",
    "vehicle.motorcycle": "MOTORBIKE",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "vehicle.bus.rigid": "BUS",
    "vehicle.bus": "BUS",
    "vehicle.construction": "CAR",
    "vehicle.trailer": "CAR",
    "movable_object.pushable_pullable": "UNKNOWN",
    "movable_object.debris": "UNKNOWN",
    "static_object.bicycle rack": "BICYCLE",
    "human.pedestrian.personal_mobility": "PEDESTRIAN",
    "pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
    "human.pedestrian.stroller": "PEDESTRIAN",
    "vehicle.bus.bendy": "BUS",
    "animal": "ANIMAL",
    "vehicle.emergency.police": "CAR",
    "vehicle.emergency.ambulance": "CAR",
    "human.pedestrian.wheelchair": "PEDESTRIAN",
    "vehicle.ego": "UNKNOWN",
    "static_object.bollard": "UNKNOWN",  # tier4 data original
}

label_to_index = {
    "vehicle.car": 0,  # "CAR",
    "vehicle.construction": 1,  # "CAR",
    "vehicle.emergency.police": 2,  # "CAR",
    "vehicle.emergency.ambulance": 3,  # "CAR",
    "vehicle.trailer": 4,  # "CAR",
    "human.pedestrian.adult": 5,  # "PEDESTRIAN",
    "human.pedestrian.child": 6,  # "PEDESTRIAN",
    "human.pedestrian.construction_worker": 7,  # "PEDESTRIAN",
    "human.pedestrian.police_officer": 8,  # "PEDESTRIAN",
    "human.pedestrian.stroller": 9,  # "PEDESTRIAN",
    "human.pedestrian.personal_mobility": 10,  # "PEDESTRIAN",
    "human.pedestrian.wheelchair": 11,  # "PEDESTRIAN",
    "vehicle.truck": 12,  # "TRUCK",
    "vehicle.bus.rigid": 13,  # "BUS",
    "vehicle.bus.bendy": 14,  # "BUS",
    "static_object.bicycle_rack": 15,  # "BICYCLE",
    "vehicle.bicycle": 16,  # "BICYCLE",
    "vehicle.motorcycle": 17,  # "MOTORBIKE",
    "animal": 18,  # "ANIMAL",
    "movable_object.pushable_pullable": 19,  # "None",
    "movable_object.debris": 20,  # "None",
    "movable_object.barrier": 21,  # "None",
    "movable_object.trafficcone": 22,  # "None",
    "vehicle.ego": 23,  # 'None',
}

id_to_nu = {v: i for i, v in label_to_index.items()}

t4label_to_index = {
    "UNKNOWN": 0,
    "CAR": 1,
    "TRUCK": 2,
    "BUS": 3,
    "BICYCLE": 4,
    "MOTORBIKE": 5,
    "PEDESTRIAN": 6,
    "ANIMAL": 7,
}
id_to_label = {v: i for i, v in t4label_to_index.items()}


def vis2(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    For demo_custom0
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
        # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)

        # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(
        #     img,
        #     (x0, y0 + 1),
        #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        #     txt_bk_color,
        #     -1
        # )
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    for cnt, (t4_label, cls_id) in enumerate(t4label_to_index.items()):
        cv2.rectangle(
            img,
            (25, 10 * (2 * cnt + 1)),
            (50, 10 * (2 * cnt + 1) + 10),
            (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
            -1,
        )
        cv2.putText(
            img,
            t4_label,
            (60, 10 * (2 * cnt + 1) + 15),
            font,
            0.4,
            (_COLORS[cls_id] * 255).astype(np.uint8).tolist(),
            thickness=1,
        )

    return img


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
