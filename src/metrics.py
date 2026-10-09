import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@torch.no_grad()
def dice_iou_per_image(seg_logits, mask, threshold=0.5):
    """Per-image Dice and IoU of the thresholded prediction. Returns two 1-D numpy arrays."""
    pred = (torch.sigmoid(seg_logits) > threshold).float()
    inter = (pred * mask).sum(dim=(1, 2, 3))
    p, t = pred.sum(dim=(1, 2, 3)), mask.sum(dim=(1, 2, 3))
    empty = (p + t) == 0  # both empty counts as perfect
    dice = torch.where(empty, torch.ones_like(inter), 2 * inter / (p + t).clamp(min=1))
    iou = torch.where(empty, torch.ones_like(inter), inter / (p + t - inter).clamp(min=1))
    return dice.cpu().numpy(), iou.cpu().numpy()


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def segmentation_metrics(dice, iou, labels):
    dice, iou, labels = map(np.asarray, (dice, iou, labels))
    out = {"dice": float(dice.mean()), "dice_std": float(dice.std()), "iou": float(iou.mean())}
    for k, name in enumerate(["meningioma", "glioma", "pituitary"]):
        if (labels == k).any():
            out[f"dice_{name}"] = float(dice[labels == k].mean())
    return out
