import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate, generate_attention_maps_ms
import models
import utils
import random
import numpy as np

from PAR import PAR


def get_args_parser():
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=60, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="deit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")

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

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # Dataset parameters
    parser.add_argument("--data-path", default="", type=str, help="dataset path")
    parser.add_argument("--img-list", default="", type=str, help="image list path")
    parser.add_argument("--data-set", default="", type=str, help="dataset")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # generating attention maps
    parser.add_argument("--gen_attention_maps", action="store_true")
    parser.add_argument("--PAR", default=0, type=int)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--attention-dir", type=str, default=None)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=12,
        help="extract attention maps from the last layers",
    )

    parser.add_argument("--patch-attn-refine", action="store_true")
    parser.add_argument("--visualize-cls-attn", action="store_true")

    parser.add_argument("--gt-dir", type=str, default=None)
    parser.add_argument("--cam-npy-dir", type=str, default=None)
    parser.add_argument("--scales", nargs="+", type=float)
    parser.add_argument("--label-file-path", type=str, default=None)
    parser.add_argument("--attention-type", type=str, default="fused")

    parser.add_argument("--out-crf", type=str, default=None)
    parser.add_argument("--low_alpha", default=1, type=int)
    parser.add_argument("--high_alpha", default=12, type=int)

    # super tokens attention layer
    parser.add_argument("--st_attn_depth", default=3, type=int)

    # for end-to-end
    parser.add_argument("--end_to_end", action="store_true")
    parser.add_argument("--bg_thr", type=float)
    parser.add_argument("--skip_seg", type=int)
    parser.add_argument("--ignore_delta", type=float)

    return parser


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    print(args)

    device = torch.device(args.device)
    same_seeds(0)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_train_, args.nb_classes = build_dataset(
        is_train=False, gen_attn=True, args=args
    )
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_train_ = torch.utils.data.DataLoader(
        dataset_train_,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=0,
        # num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        st_attn_depth=args.st_attn_depth,
    )

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        try:
            checkpoint_model = checkpoint["model"]
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        if args.finetune.startswith("https"):
            num_extra_tokens = 1
        else:
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)

        new_size = int(num_patches**0.5)

        if args.finetune.startswith("https") and "MCTformer" in args.model:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(
                1, args.nb_classes, 1
            )
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

        if args.finetune.startswith("https") and "MCTformer" in args.model:
            cls_token_checkpoint = checkpoint_model["cls_token"]
            new_cls_token = cls_token_checkpoint.repeat(1, args.nb_classes, 1)
            checkpoint_model["cls_token"] = new_cls_token

            for st_attn_idx in range(12 - args.st_attn_depth, 12):
                # ln_pat
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.bias"]
                # ln_pat2
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat2.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat2.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.bias"]
                # qkv
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.qkv.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.qkv.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.qkv.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.qkv.bias"]
                # proj
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.proj.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.proj.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.proj.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.proj.bias"]
                # ln_cls
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_cls.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_cls.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.bias"]
                # ln_pat2
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat2.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_pat2.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.bias"]
                # ln_st
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_st.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.ln_st.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.norm1.bias"]
                # qkv_stsa
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.qkv_stsa.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.qkv.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.qkv_stsa.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.qkv.bias"]
                # proj_cls
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.proj_cls.weight"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.proj.weight"]
                checkpoint_model[
                    f"blocks.{st_attn_idx}.st_attn.proj_cls.bias"
                ] = checkpoint_model[f"blocks.{st_attn_idx}.attn.proj.bias"]

        print("Modules not in checkpoint:")
        for m in model.state_dict().keys():
            if m not in checkpoint_model.keys():
                print("\t", m)
        print("\nSkip (not used)")
        for m in checkpoint_model.keys():
            if m not in model.state_dict().keys():
                print("\t", m)
        print()

        model.load_state_dict(checkpoint_model, strict=False)

    model = torch.nn.DataParallel(model).cuda()
    # model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%"
        )
        return

    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        generate_attention_maps_ms(data_loader_train_, model, device, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

    for epoch in range(args.start_epoch, args.epochs):
        if epoch < args.skip_seg:
            skip_seg = True
        else:
            skip_seg = False

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            scales=args.scales,
            bg_thr=args.bg_thr,
            par=par,
            skip_seg=skip_seg,
            ignore_delta=args.ignore_delta,
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, skip_seg=skip_seg)
        print(
            f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%"
        )
        if test_stats["mAP"] > max_accuracy and args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint_best.pth"]
            for checkpoint_path in checkpoint_paths:
                torch.save({"model": model.state_dict()}, checkpoint_path)

        if not skip_seg:
            checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
            for checkpoint_path in checkpoint_paths:
                torch.save({"model": model.state_dict()}, checkpoint_path)

        max_accuracy = max(max_accuracy, test_stats["mAP"])
        print(f"Max mAP: {max_accuracy * 100:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    torch.save({"model": model.state_dict()}, output_dir / "checkpoint.pth")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DeiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
