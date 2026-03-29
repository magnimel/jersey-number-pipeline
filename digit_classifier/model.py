import torch.nn as nn


class DigitCountMoviNet(nn.Module):
    def __init__(self, model_id="a0"):
        super().__init__()
        from movinets import MoViNet
        from movinets.config import _C

        cfg_map = {
            "a0": _C.MODEL.MoViNetA0,
            "a1": _C.MODEL.MoViNetA1,
            "a2": _C.MODEL.MoViNetA2,
            "a3": _C.MODEL.MoViNetA3,
            "a4": _C.MODEL.MoViNetA4,
            "a5": _C.MODEL.MoViNetA5,
        }
        if model_id not in cfg_map:
            raise ValueError(f"model_id must be one of {list(cfg_map)}, got '{model_id}'")
        self.backbone = MoViNet(cfg_map[model_id], causal=False, pretrained=True)
        # Freeze the entire backbone — only the classifier head is fine-tuned
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Replace last ConvBlock3D in classifier with a plain Conv3d for binary output
        last = self.backbone.classifier[-1]
        in_ch = last.conv_1[0].in_channels
        self.backbone.classifier[-1] = nn.Conv3d(in_ch, 1, (1, 1, 1))
        # Unfreeze classifier head
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, video):
        # video: (B, C, T, H, W) -> out: (B, 1, 1, 1, 1)
        return self.backbone(video).flatten(1).squeeze(-1)  # (B,)
