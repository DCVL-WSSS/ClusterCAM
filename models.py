import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

import math


__all__ = ["deit_small_MCTformerV2_patch16_224"]


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    "3 x 3 conv"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    "1 x 1 conv"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class LargeFOV(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=5):
        super(LargeFOV, self).__init__()
        self.embed_dim = 512
        self.dilation = dilation
        self.conv6 = conv3x3(
            in_planes=in_planes,
            out_planes=self.embed_dim,
            padding=self.dilation,
            dilation=self.dilation,
        )
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(
            in_planes=self.embed_dim,
            out_planes=self.embed_dim,
            padding=self.dilation,
            dilation=self.dilation,
        )
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = conv1x1(in_planes=self.embed_dim, out_planes=out_planes, padding=0)

    def _init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        return None

    def forward(self, x):
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        out = self.conv8(x)

        return out


class MCTformerV2(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(
            self.embed_dim + 12 * 6 * self.num_classes,  # dim + 12 * Head * K
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.decoder = LargeFOV(
            in_planes=self.embed_dim + 12 * 6 * self.num_classes,  # dim + 12 * Head * K
            out_planes=self.num_classes + 1,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_classes, self.embed_dim)
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0 : self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes :]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        p2p_list = []
        c2p_list = []

        for i, blk in enumerate(self.blocks):
            x, p2p, c2p = blk(x, w0, h0)
            p2p_list.append(p2p)
            if c2p is not None:
                c2p_list.append(c2p)

        return (
            x[:, 0 : self.num_classes],
            x[:, self.num_classes :],
            p2p_list,
            c2p_list,
        )

    def forward(self, x, cam_only=False, skip_seg=False, n_layers=3, p2p_refine=True):
        w, h = x.shape[2:]
        x_cls, x_patch, p2p_attn_list, c2p_attn_list = self.forward_features(x)
        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])

        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()  # (B, dim, H, W)
        c2p_attn = torch.stack(c2p_attn_list)  # (12, B, Head, K, HW)
        c2p_feat = c2p_attn.permute(1, 0, 2, 3, 4)
        c2p_feat = c2p_feat.reshape(n, -1, x_patch.shape[2], x_patch.shape[3])
        # c2p_attn = (B, 12*Head*K, H, W)
        x_patch = torch.concat([x_patch, c2p_feat], dim=1)  # (B, dim + 12*Head*K, H, W)

        if cam_only:
            cams = F.conv2d(x_patch, self.head.weight, stride=1, padding=1).detach()
            c2p_attn = torch.mean(c2p_attn, dim=2)  # (12, B, K, HW)

            # feature_map = x_patch1.detach().clone()  # (B, C, 14, 14)
            cams = F.relu(cams)

            n, c, h, w = cams.shape

            mtatt = c2p_attn[-n_layers:].sum(0).reshape([n, c, h, w])
            cams = mtatt * cams  # (B, K, 14, 14)

            if p2p_refine:
                p2p_attn = torch.stack(p2p_attn_list)  # (12, B, H, HW, HW)
                p2p_attn = torch.mean(p2p_attn, dim=2)  # (12, B, HW, HW)
                p2p_attn = torch.sum(p2p_attn, dim=0)  # (B, HW, HW)

                cams = torch.matmul(
                    p2p_attn.unsqueeze(1),
                    cams.view(cams.shape[0], cams.shape[1], -1, 1),
                ).reshape(cams.shape)

            return cams

        segs = None
        if not skip_seg:
            segs = self.decoder(x_patch)

        cls_logits = x_cls.mean(-1)
        x_patch = self.head(x_patch)  # (B, K, H, W)
        pat_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        return cls_logits, pat_logits, segs

        # x_patch = self.head(x_patch)  # (B, K, H, W)

        # c2p_attn = torch.mean(c2p_attn, dim=2)  # (12, B, K, HW)

        # x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        # p2p_attn = torch.stack(p2p_attn_list)  # (12, B, H, HW, HW)
        # p2p_attn = torch.mean(p2p_attn, dim=2)  # (12, B, HW, HW)

        # feature_map = x_patch.detach().clone()  # (B, C, 14, 14)
        # feature_map = F.relu(feature_map)

        # n, c, h, w = feature_map.shape

        # mtatt = c2p_attn[-n_layers:].sum(0).reshape([n, c, h, w])

        # if attention_type == "fused":
        #     cams = mtatt * feature_map  # (B, C, 14, 14)
        # elif attention_type == "patchcam":
        #     cams = feature_map
        # else:
        #     cams = mtatt

        # x_cls_logits = x_cls.mean(-1)

        # return x_cls_logits, x_patch_logits


@register_model
def deit_small_MCTformerV2_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV2(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )["model"]
        model_dict = model.state_dict()
        for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k not in ["cls_token", "pos_embed"]
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
