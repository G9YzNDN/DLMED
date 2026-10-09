"""Inference loop shared by train.py (validation) and evaluate.py (test)."""
import numpy as np
import torch

from metrics import classification_metrics, dice_iou_per_image, segmentation_metrics


@torch.no_grad()
def predict(model, loader, device):
    """Run the model over a loader. Returns per-image results and summary metrics for both heads."""
    model.eval()
    ids, labels, preds, probs, dices, ious = [], [], [], [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        seg_logits, cls_logits = model(x)
        d, j = dice_iou_per_image(seg_logits, batch["mask"].to(device))
        p = torch.softmax(cls_logits, dim=1).cpu().numpy()
        ids += list(batch["id"])
        labels += batch["label"].tolist()
        preds += p.argmax(1).tolist()
        probs.append(p)
        dices.append(d)
        ious.append(j)
    per_image = {
        "id": ids, "label": labels, "pred": preds,
        "probs": np.concatenate(probs), "dice": np.concatenate(dices), "iou": np.concatenate(ious),
    }
    metrics = {
        "cls": classification_metrics(labels, preds),
        "seg": segmentation_metrics(per_image["dice"], per_image["iou"], labels),
    }
    return per_image, metrics


def selection_score(metrics, tasks):
    """Validation score used to pick the best epoch."""
    if tasks == "cls":
        return metrics["cls"]["macro_f1"]
    if tasks == "seg":
        return metrics["seg"]["dice"]
    return (metrics["cls"]["macro_f1"] + metrics["seg"]["dice"]) / 2
