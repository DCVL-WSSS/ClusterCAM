import numpy as np
import torch
from torch.backends import cudnn

cudnn.enabled = True
import importlib
from seg.tool import imutils
import argparse
import cv2
import os.path
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from seg.tool.metrics import Evaluator
import PIL.Image as Image

import models
from timm.models import create_model


palette = [
    0,
    0,
    0,
    128,
    0,
    0,
    0,
    128,
    0,
    128,
    128,
    0,
    0,
    0,
    128,
    128,
    0,
    128,
    0,
    128,
    128,
    128,
    128,
    128,
    64,
    0,
    0,
    192,
    0,
    0,
    64,
    128,
    0,
    192,
    128,
    0,
    64,
    0,
    128,
    192,
    0,
    128,
    64,
    128,
    128,
    192,
    128,
    128,
    0,
    64,
    0,
    128,
    64,
    0,
    0,
    192,
    0,
    128,
    192,
    0,
    0,
    64,
    128,
    128,
    64,
    128,
    0,
    192,
    128,
    128,
    192,
    128,
    64,
    64,
    0,
    192,
    64,
    0,
    64,
    192,
    0,
    192,
    192,
    0,
]

classes = np.array(
    (
        "background",  # always index 0
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
)


def crf_postprocess(pred_prob, ori_img):
    crf_score = imutils.crf_inference_inf(ori_img, pred_prob, labels=21)
    return crf_score


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1", "True"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", "False"):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="", type=str)
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--network", default="", type=str)
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--save_path_c", default=None, type=str)
    parser.add_argument("--list_path", default="./voc12/val_id.txt", type=str)
    parser.add_argument("--img_path", default="", type=str)
    parser.add_argument("--num_classes", default=21, type=int)
    parser.add_argument("--use_crf", default=False, type=str2bool)
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--st_attn_depth", default=12, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    args = parser.parse_args()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if not args.test:
        Path(args.save_path_c).mkdir(parents=True, exist_ok=True)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes - 1,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        st_attn_depth=args.st_attn_depth,
    )

    # model = getattr(importlib.import_module("network." + args.network), "Net")(
    #     num_classes=args.num_classes
    # )

    ckpt = torch.load(args.weights)["model"]

    from collections import OrderedDict

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_ckpt[name] = v

    model.load_state_dict(new_ckpt)
    model = torch.nn.DataParallel(model).cuda()

    seg_evaluator = Evaluator(num_class=args.num_classes)
    model.eval()
    # model.cuda()
    im_path = args.img_path
    img_list = open(args.list_path).readlines()

    with torch.no_grad():
        for idx in tqdm(range(len(img_list))):
            i = img_list[idx]

            img_temp = cv2.imread(os.path.join(im_path, i.strip() + ".jpg"))
            img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float64)
            img_original = img_temp.astype(np.uint8)

            img_temp[:, :, 0] = (img_temp[:, :, 0] / 255.0 - 0.485) / 0.229
            img_temp[:, :, 1] = (img_temp[:, :, 1] / 255.0 - 0.456) / 0.224
            img_temp[:, :, 2] = (img_temp[:, :, 2] / 255.0 - 0.406) / 0.225

            input = (
                torch.from_numpy(img_temp[np.newaxis, :].transpose(0, 3, 1, 2))
                .float()
                .cuda()
            )

            N, C, H, W = input.size()

            probs = torch.zeros((N, args.num_classes, H, W)).cuda()
            if args.scales:
                scales = tuple(args.scales)

            for s in scales:
                new_hw = [int(H * s), int(W * s)]
                im = F.interpolate(input, new_hw, mode="bilinear", align_corners=True)
                cls_logits, pat_logits, prob = model(x=im)

                prob = F.interpolate(prob, (H, W), mode="bilinear", align_corners=False)
                prob = F.softmax(prob, dim=1)
                probs = torch.max(probs, prob)

            output = probs.cpu().data[0].numpy()

            if args.use_crf:
                crf_output = crf_postprocess(output, img_original)
                pred = np.argmax(crf_output, 0)
            else:
                pred = np.argmax(output, axis=0)

            if not args.test:
                gt = Image.open(os.path.join(args.gt_path, i.strip() + ".png"))
                gt = np.asarray(gt)
                seg_evaluator.add_batch(gt, pred)

            save_path = os.path.join(args.save_path, i.strip() + ".png")
            cv2.imwrite(save_path, pred.astype(np.uint8))

            if args.save_path_c:
                out = pred.astype(np.uint8)
                out = Image.fromarray(out, mode="P")
                out.putpalette(palette)
                out_name = os.path.join(args.save_path_c, i.strip() + ".png")
                out.save(out_name)

        if args.test:
            exit()

        IoU, mIoU = seg_evaluator.Mean_Intersection_over_Union()

        str_format = "{:<15s}\t{:<15.2%}"
        filename = os.path.join(args.save_path, "result.txt")
        with open(filename, "w") as f:
            for k in range(args.num_classes):
                print(str_format.format(classes[k], IoU[k]))
                f.write(
                    "class {:2d} {:12} IU {:.3f}".format(k, classes[k], IoU[k]) + "\n"
                )
            f.write("mIoU = {:.3f}".format(mIoU) + "\n")
        print(f"mIoU={mIoU:.3f}")
