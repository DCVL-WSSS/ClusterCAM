import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math

import numpy as np


def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes**2,
    )
    return hist.reshape(num_classes, num_classes)


def eval_scores(label_trues, label_preds, num_classes=21):
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.numpy().flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    _acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(_acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "pAcc": acc,
        "mAcc": acc_cls,
        "miou": mean_iu,
        "iou": cls_iu,
    }


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(
        pred, bg_label.type(torch.long), ignore_index=ignore_index
    )
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(
        pred, fg_label.type(torch.long), ignore_index=ignore_index
    )

    if not math.isfinite(fg_loss):
        fg_loss = 0
    if not math.isfinite(bg_loss):
        bg_loss = 0

    return (bg_loss + fg_loss) * 0.5


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(
        refined_cams, size=orig_size, mode="bilinear", align_corners=False
    )
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def refine_cams_with_bkg_v2(
    ref_mod=None,
    images=None,
    cams=None,
    cls_labels=None,
    high_thre=None,
    low_thre=None,
    ignore_index=False,
    img_box=None,
    down_scale=2,
):
    b, _, h, w = images.shape
    _images = F.interpolate(
        images,
        size=[h // down_scale, w // down_scale],
        mode="bilinear",
        align_corners=False,
    )

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(
        cams_with_bkg_h,
        size=[h // down_scale, w // down_scale],
        mode="bilinear",
        align_corners=False,
    )  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(
        cams_with_bkg_l,
        size=[h // down_scale, w // down_scale],
        mode="bilinear",
        align_corners=False,
    )  # .softmax(dim=1)

    for idx, _ in enumerate(images):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(
            ref_mod=ref_mod,
            images=_images[[idx], ...],
            cams=valid_cams_h,
            valid_key=valid_key,
            orig_size=(h, w),
        )
        _refined_label_l = _refine_cams(
            ref_mod=ref_mod,
            images=_images[[idx], ...],
            cams=valid_cams_l,
            valid_key=valid_key,
            orig_size=(h, w),
        )

        refined_label_h[idx] = _refined_label_h[0]
        refined_label_l[idx] = _refined_label_l[0]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def cam_to_label(
    cam,
    cls_label,
    # img_box=None,
    bkg_thre=None,
    high_thre=None,
    low_thre=None,
    ignore_mid=False,
    ignore_index=None,
):
    b, c, h, w = cam.shape

    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    # if img_box is None:
    #     return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    # for idx, coord in enumerate(img_box):
    #     pseudo_label[idx, coord[0] : coord[1], coord[2] : coord[3]] = _pseudo_label[
    #         idx, coord[0] : coord[1], coord[2] : coord[3]
    #     ]

    return valid_cam, pseudo_label


def multi_scale_cam2(model, inputs, scales):
    """process cam and aux-cam"""
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode="bilinear", align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(
                    inputs,
                    size=(int(s * h), int(s * w)),
                    mode="bilinear",
                    align_corners=False,
                )
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(
                    _cam, size=(h, w), mode="bilinear", align_corners=False
                )
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    return cam


def denormalize_img(
    imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
):
    _imgs = torch.zeros_like(imgs)
    _imgs[:, 0, :, :] = imgs[:, 0, :, :] * std[0] + mean[0]
    _imgs[:, 1, :, :] = imgs[:, 1, :, :] * std[1] + mean[1]
    _imgs[:, 2, :, :] = imgs[:, 2, :, :] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs


def denormalize_img2(imgs=None):
    # _imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
