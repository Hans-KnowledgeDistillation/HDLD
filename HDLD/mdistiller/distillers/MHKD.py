from termios import CEOL
from turtle import st
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ._base import Distiller
from .loss import CrossEntropyLabelSmooth

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def cat_mask(t, mask1, mask2, mask3, mask4, mask5, not_mask):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    t3 = (t * mask3).sum(dim=1, keepdims=True)
    t4 = (t * mask4).sum(dim=1, keepdims=True)
    t5 = (t * mask5).sum(dim=1, keepdims=True)
    tn = (t * not_mask).sum(1, keepdims=True)
    rt = torch.cat([t1, t2, t3, t4, t5, tn], dim=1)
    return rt

def top_not_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.ones_like(logits).scatter_(1, pred_s_index, 0).bool()
    return mask

def top_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.zeros_like(logits).scatter_(1, pred_s_index, 1).bool()
    return mask

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def hc_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand=False):
    y_s = normalize(logits_student_in) if logit_stand else logits_student_in
    y_t = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    pred_student = F.softmax(y_s / temperature, dim=1)
    pred_teacher = F.softmax(y_t / temperature, dim=1)
    #s_mask_t = _get_gt_mask(y_s, target)

    s_mask_5 = top_mask(y_t, 5).int() - top_mask(y_t, 4).int()
    s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
    s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
    s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
    s_mask_1 = top_mask(y_t,1).int()
    s_mask = top_mask(y_t, 5)
    not_s_mask = top_not_mask(y_t, 5)
    pred_student = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)
    pred_teacher = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)

    log_pred_student = torch.log(pred_student)
    loss_top7 = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        y_t / temperature - 1000 * s_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        y_s / temperature - 1000 * s_mask, dim=1
    )

    not_loss_top7 = F.kl_div(log_pred_student_part2, pred_teacher_part2,
                             size_average=False) * (temperature ** 2) / target.size()[0]

    return (alpha * loss_top7 + beta * not_loss_top7) / 5


def cc_loss(y_s, y_t, alpha, beta, temperature, reduce=True):
    batch_size, class_num = y_t.shape

    pred_student = F.softmax(y_s / temperature, dim=1)
    pred_teacher = F.softmax(y_t / temperature, dim=1)
    # s_mask_t = _get_gt_mask(y_s, target)

    s_mask_5 = top_mask(y_t, 5).int() - top_mask(y_t, 4).int()
    s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
    s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
    s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
    s_mask_1 = top_mask(y_t, 1).int()
    s_mask = top_mask(y_t, 5)
    not_s_mask = top_not_mask(y_t, 5)
    pred_student_h = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)
    pred_teacher_h = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)

    pred_teacher_t = F.softmax(
        y_t / temperature - 1000 * s_mask, dim=1
    )
    pred_student_t = F.softmax(
        y_s / temperature - 1000 * s_mask, dim=1
    )

    student_matrix = torch.mm(pred_student_h.transpose(1, 0), pred_student_h)
    teacher_matrix = torch.mm(pred_teacher_h.transpose(1, 0), pred_teacher_h)
    student_matrix1 = torch.mm(pred_student_t.transpose(1, 0), pred_student_t)
    teacher_matrix1 = torch.mm(pred_teacher_t.transpose(1, 0), pred_teacher_t)
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha / 5 + (
                   (teacher_matrix1 - student_matrix1) ** 2).sum() * beta / 5) / class_num
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha / 5 + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta / 5) / class_num
    return consistency_loss


def bc_loss(y_s, y_t, alpha, beta, temperature, reduce=True):
    batch_size, class_num = y_t.shape

    pred_student = F.softmax(y_s / temperature, dim=1)
    pred_teacher = F.softmax(y_t / temperature, dim=1)
    # s_mask_t = _get_gt_mask(y_s, target)

    s_mask_5 = top_mask(y_t, 5).int() - top_mask(y_t, 4).int()
    s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
    s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
    s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
    s_mask_1 = top_mask(y_t, 1).int()
    s_mask = top_mask(y_t, 5)
    not_s_mask = top_not_mask(y_t, 5)
    pred_student_h = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)
    pred_teacher_h = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)

    pred_teacher_t = F.softmax(
        y_t / temperature - 1000 * s_mask, dim=1
    )
    pred_student_t = F.softmax(
        y_s / temperature - 1000 * s_mask, dim=1
    )

    student_matrix = torch.mm(pred_student_h, pred_student_h.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher_h, pred_teacher_h.transpose(1, 0))
    student_matrix1 = torch.mm(pred_student_t, pred_student_t.transpose(1, 0))
    teacher_matrix1 = torch.mm(pred_teacher_t, pred_teacher_t.transpose(1, 0))
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha / 5 + (
                   (teacher_matrix1 - student_matrix1) ** 2).sum() * beta / 5) / batch_size
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha / 5 + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta / 5) / batch_size
    return consistency_loss

class MHKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(MHKD, self).__init__(student, teacher)
        self.temperature = cfg.DKD.T
        self.warmup = cfg.HDLD.WARMUP
        self.alpha = cfg.HDLD.ALPHA
        self.beta = cfg.HDLD.BETA
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

    def forward_train(self, image_weak, image_strong, target, **kwargs):
        logits_student_weak, _ = self.student(image_weak)
        logits_student_strong, _ = self.student(image_strong)
        with (torch.no_grad()):
            logits_teacher_weak, _ = self.teacher(image_weak)
            logits_teacher_strong, _ = self.teacher(image_strong)

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 50
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 50
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.ce_loss_weight * (
                    F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))

        loss_kd_weak = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * ((hc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((hc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            3.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((hc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            5.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((hc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()) + self.kd_loss_weight * ((hc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            logit_stand=self.logit_stand,
        ) * mask).mean()))

        loss_kd_strong = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * hc_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * hc_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            3.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * hc_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            5.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * hc_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            2.0,
            logit_stand=self.logit_stand,
        ) + self.kd_loss_weight * hc_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            6.0,
            logit_stand=self.logit_stand,
        ))

        loss_cc_weak = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * ((cc_loss(
            # loss_cc_weak= self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            self.temperature,
            # reduce=False
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            3.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            5.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            2.0,
        ) * class_conf_mask).mean()) + self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            6.0,
        ) * class_conf_mask).mean()))

        loss_bc_weak = min(kwargs["epoch"] / self.warmup, 1.0) * (self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            self.temperature,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            3.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            5.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            2.0,
        ) * mask).mean()) + self.kd_loss_weight * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            self.alpha,
            self.beta,
            6.0,
        ) * mask).mean()))

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd_weak + loss_kd_strong,
            #"loss_cc": loss_cc_weak,
            #"loss_bc": loss_bc_weak,
        }
        return logits_student_weak, losses_dict

