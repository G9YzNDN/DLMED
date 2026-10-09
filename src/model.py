"""Shared-encoder multi-task model: U-Net decoder (segmentation) + classification head.

The same architecture is used for every experiment. Single-task baselines are trained by
switching one loss off (see train.py --tasks), so encoder, data and training settings
are identical across experiments.
"""
import segmentation_models_pytorch as smp

NUM_CLASSES = 3


def build_model(encoder="resnet34", pretrained=True):
    # aux_params adds a classification head on the deepest encoder feature map.
    # forward(x) -> (seg_logits (B,1,H,W), cls_logits (B,3))
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=1,
        aux_params={"classes": NUM_CLASSES, "dropout": 0.2},
    )
