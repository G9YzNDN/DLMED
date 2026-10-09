import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def seg_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def cls_loss(logits, labels):
    return F.cross_entropy(logits, labels)


class MultiTaskLoss(nn.Module):
    """Combines the two task losses.

    weighting="fixed":       L = w_seg * L_seg + w_cls * L_cls
    weighting="uncertainty": L = sum_i exp(-s_i) * L_i + s_i, with learnable s_i
                             (Kendall et al., CVPR 2018). Its parameters must be optimized.
    Single-task baselines: set the other task's weight to 0 (fixed weighting only).
    """

    def __init__(self, weighting="fixed", w_seg=1.0, w_cls=1.0):
        super().__init__()
        self.weighting = weighting
        self.w_seg, self.w_cls = w_seg, w_cls
        if weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, seg_logits, cls_logits, mask, label):
        l_seg = seg_loss(seg_logits, mask)
        l_cls = cls_loss(cls_logits, label)
        if self.weighting == "uncertainty":
            s = self.log_vars
            total = torch.exp(-s[0]) * l_seg + s[0] + torch.exp(-s[1]) * l_cls + s[1]
        else:
            total = self.w_seg * l_seg + self.w_cls * l_cls
        return total, {"loss_seg": l_seg.item(), "loss_cls": l_cls.item()}
