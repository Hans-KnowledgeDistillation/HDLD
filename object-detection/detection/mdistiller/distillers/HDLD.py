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
    # t6 = (t * mask6).sum(dim=1, keepdims=True)
    # t7 = (t * mask7).sum(dim=1, keepdims=True)
    tn = (t * not_mask).sum(1, keepdims=True)
    rt = torch.cat([t1, t2, t3, tn], dim=1)
    #rt = torch.cat([t1, t2, t3, t4, t5, tn], dim=1)
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
    s_mask_t = _get_gt_mask(y_s, target)

    #s_mask_7 = top_mask(y_t, 7).int() - top_mask(y_t, 6).int()
    #s_mask_6 = top_mask(y_t, 6).int() - top_mask(y_t, 5).int()
    s_mask_5 = top_mask(y_t, 5).int() - top_mask(y_t, 4).int()
    s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
    s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
    s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
    s_mask_1 = top_mask(y_t,1).int()
    s_mask = top_mask(y_t, 3)
    not_s_mask = top_not_mask(y_t, 3)
    # pred_student = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, s_mask_6, s_mask_7, not_s_mask)
    # pred_teacher = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, s_mask_6, s_mask_7, not_s_mask)
    pred_student = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5,
                            not_s_mask)
    pred_teacher = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5,
                            not_s_mask)

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

    return alpha * loss_top7 + beta * not_loss_top7

def hc_loss1(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand=False):
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

    return alpha * loss_top7 + beta * not_loss_top7

class HDLD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(HDLD, self).__init__(student, teacher)
        self.temperature = cfg.HDLD.T
        self.warmup = cfg.HDLD.WARMUP
        self.alpha = cfg.HDLD.ALPHA
        self.beta = cfg.HDLD.BETA
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * self.kd_loss_weight * hc_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            logit_stand=self.logit_stand,
        )

        # loss_patch3 = min(kwargs["epoch"] / self.warmup, 1.0) * self.kd_loss_weight * multi(
        #     patch_s3,
        #     patch_t3,
        #     target,
        #     self.temperature,
        #     logit_stand=self.logit_stand,
        # )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            #"loss_patch3": loss_patch3,
        }
        return logits_student, losses_dict
