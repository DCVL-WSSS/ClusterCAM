import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path

from seg.tool.metrics import Evaluator


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    set_training_mode=True,
    scales=[1.0],
    bg_thr=0.440,  # FIXME:
    par=None,
    skip_seg=False,
    ignore_delta=0.1,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    assert scales[0] == 1.0, "first value of 'scales' must be 1.0"

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)

        samples_denorm = utils.denormalize_img2(samples.clone())
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        with torch.cuda.amp.autocast():
            if not skip_seg:
                # CAM 생성
                cams = utils.multi_scale_cam2(model, inputs=samples, scales=scales)

                # PGT 생성
                valid_cam, _ = utils.cam_to_label(
                    cams.detach(),
                    cls_label=targets,
                    ignore_mid=True,
                    bkg_thre=bg_thr,
                    high_thre=bg_thr + ignore_delta,
                    low_thre=bg_thr - ignore_delta,
                    ignore_index=255,
                )
                refined_pseudo_label = utils.refine_cams_with_bkg_v2(
                    par,
                    samples_denorm,
                    cams=valid_cam,
                    cls_labels=targets,
                    high_thre=bg_thr + ignore_delta,
                    low_thre=bg_thr - ignore_delta,
                    ignore_index=255,
                    # img_box=img_box,
                )

            # MODEL forward
            cls_logits, pat_logits, segs = model(samples, skip_seg=skip_seg)

            # Loss 계산

            closs = F.multilabel_soft_margin_loss(cls_logits, targets)
            metric_logger.update(cls_loss=closs.item())

            ploss = F.multilabel_soft_margin_loss(pat_logits, targets)
            metric_logger.update(pat_loss=ploss.item())

            if not skip_seg:
                segs = F.interpolate(
                    segs,
                    size=refined_pseudo_label.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )
                sloss = utils.get_seg_loss(
                    segs,
                    refined_pseudo_label.type(torch.long),
                    ignore_index=255,
                )
                metric_logger.update(seg_loss=sloss.item())
                loss = closs + ploss + 0.1 * sloss
            else:
                loss = closs + ploss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, skip_seg=False):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode

    model.eval()

    seg_evaluator = Evaluator(num_class=21)

    for images, target, seg_label in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images, skip_seg=skip_seg)
            if not isinstance(output, torch.Tensor):
                cls_logits, pat_logits, segs = output
            loss = criterion(cls_logits, target)
            output = torch.sigmoid(cls_logits)

            if not skip_seg:
                segs = F.interpolate(segs, seg_label.shape[1:], mode="nearest")
                segs = torch.argmax(segs, 1)
                seg_label = seg_label.cpu().numpy()
                segs = segs.cpu().numpy()
                seg_evaluator.add_batch(seg_label, segs)

            # if not skip_seg:
            #     resized_segs = F.interpolate(
            #         segs, size=seg_label.shape[1:], mode="bilinear", align_corners=False
            #     )
            #     pred = torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16)
            #     gts += list(seg_label)
            #     preds += list(pred)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters["mAP"].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # if not skip_seg:
    #     seg_score = utils.eval_scores(gts, preds)
    #     metric_logger.update(seg_miou=seg_score["miou"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if not skip_seg:
        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        metric_logger.update(mIoU=mIoU)
        print(
            "* mAP {mAP.global_avg:.3f} mIoU {mIoU.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                mAP=metric_logger.mAP,
                mIoU=metric_logger.mIoU,
                losses=metric_logger.loss,
            )
        )
    else:
        print(
            "* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                mAP=metric_logger.mAP, losses=metric_logger.loss
            )
        )
    # if not skip_seg:
    #     print(
    #         "* seg mIoU : {seg_miou.global_avg:.3f}".format(
    #             seg_miou=metric_logger.seg_miou
    #         )
    #     )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generating attention maps:"
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(os.path.join(args.img_list, "train_aug_id.txt")).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
        # for iter, (image_list, target) in enumerate(data_loader):
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.0
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.0
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.0

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size

        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = (
                    images.shape[2] - images.shape[2] % args.patch_size,
                    images.shape[3] - images.shape[3] % args.patch_size,
                )
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                # NOTE:
                cams = model(images, cam_only=True)
                # patch_attn = torch.sum(patch_attn, dim=0)

                # if args.patch_attn_refine:
                #     cls_attentions = torch.matmul(
                #         patch_attn.unsqueeze(1),
                #         cls_attentions.view(
                #             cls_attentions.shape[0], cls_attentions.shape[1], -1, 1
                #         ),
                #     ).reshape(cls_attentions.shape)

                cams = F.interpolate(
                    cams,
                    size=(w_orig, h_orig),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                cams = (
                    cams.cpu().numpy()
                    * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()
                )

                if s % 2 == 1:
                    cams = np.flip(cams, axis=-1)
                cam_list.append(cams)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            # sum_cam = sum_cam + F.adaptive_max_pool2d(-sum_cam, (1, 1))
            # sum_cam /= F.adaptive_max_pool2d(sum_cam, (1, 1)) + 1e-5

            # if args.PAR:
            #     sum_cam = par(images1, sum_cam)

            # output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b, cls_ind] > 0:
                            cls_score = 0

                            cls_attention = sum_cam[b, cls_ind, :]

                            cls_attention = (cls_attention - cls_attention.min()) / (
                                cls_attention.max() - cls_attention.min() + 1e-8
                            )
                            cls_attention = cls_attention.cpu().numpy()

                            cam_dict[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(
                                    args.attention_dir,
                                    img_name
                                    + "_"
                                    + str(cls_ind)
                                    + "_"
                                    + str(cls_score)
                                    + ".png",
                                )
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(
                            os.path.join(args.cam_npy_dir, img_name + ".npy"), cam_dict
                        )

                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order="C")
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ("_%s" % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            np.save(os.path.join(folder, img_name + ".npy"), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference

    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)
