import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T, weight=None):
        super(KDLoss, self).__init__()
        self.T = T
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, out_s, out_t, target, alpha):
        kd = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                      F.softmax(out_t / self.T, dim=1),
                      reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        F.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd


class BKDLoss(nn.Module):

    def __init__(self, cls_num_list, T, weight=None):
        super(BKDLoss, self).__init__()
        self.T = T
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss().cuda()

    def forward(self, out_s, out_t, target, alpha):
        pred_t = F.softmax(out_t/self.T, dim=1)
        if self.weight is not None:
            pred_t = pred_t * self.weight
            pred_t = pred_t / pred_t.sum(1)[:, None]
        kd = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
                        pred_t,
                        reduction='none').mean(dim=0)
        kd_loss = kd.sum() * self.T * self.T
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd
###################

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class LSKDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T, weight=None):
        super(LSKDLoss, self).__init__()
        self.T = T
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, out_s, out_t, target, alpha):
        kd = F.kl_div(F.log_softmax(normalize(out_s) / self.T, dim=1),
                      F.softmax(normalize(out_t) / self.T, dim=1),
                      reduction='none').mean(dim=0)
        kd_loss = F.kl_div(F.log_softmax(normalize(out_s)/self.T, dim=1),
                        F.softmax(normalize(out_t)/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd

# class DKDLoss(nn.Module):
#     '''
#     Distilling the Knowledge in a Neural Network
#     https://arxiv.org/pdf/1503.02531.pdf
#     '''
#     def __init__(self, cls_num_list, T, weight=None):
#         super(DKDLoss, self).__init__()
#         self.T = T + 2
#         # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
#         self.weight = weight
#         self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
#         self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()
#
#     def forward(self, out_s, out_t, target, alpha):
#         gt_mask = _get_gt_mask(out_s, target)
#         other_mask = _get_other_mask(out_t, target)
#         pred_student = F.softmax(out_s / self.T, dim=1)
#         pred_teacher = F.softmax(out_t / self.T, dim=1)
#         pred_student = cat_mask(pred_student, gt_mask, other_mask)
#         pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
#         log_pred_student = torch.log(pred_student)
#         #log_pred_student = pred_student
#         tckd = F.kl_div(log_pred_student, pred_teacher, size_average=False, reduction='none')
#         tckd = tckd.mean(dim=0) / target.shape[0]
#         pred_teacher_part2 = F.softmax(
#             out_t / self.T - 1000.0 * gt_mask, dim=1
#         )
#         log_pred_student_part2 = F.log_softmax(
#             out_s / self.T - 1000.0 * gt_mask, dim=1
#         )
#         nckd = (
#                 F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False, reduction='none')
#                 .mean(dim=0)
#                 / target.shape[0]
#         )
#         tckd_loss = (
#                 F.kl_div(log_pred_student, pred_teacher, size_average=False, reduction='batchmean')
#                 * (self.T ** 2)
#                 / target.shape[0]
#         )
#         pred_teacher_part2 = F.softmax(
#             out_t / self.T - 1000.0 * gt_mask, dim=1
#         )
#         log_pred_student_part2 = F.log_softmax(
#             out_s / self.T - 1000.0 * gt_mask, dim=1
#         )
#         nckd_loss = (
#                 F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False, reduction='batchmean')
#                 * (self.T ** 2)
#                 / target.shape[0]
#         )
#         ce_loss = self.CELoss(out_s, target)
#         kd = 3 * tckd + 8 * nckd
#         loss = alpha * (3 * tckd_loss + 8 * nckd_loss) + ce_loss
#
#         return loss, kd

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

class HDLDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T, weight=None):
        super(HDLDLoss, self).__init__()
        self.T = T
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, y_s, y_t, target, alpha):
        pred_student = F.softmax(y_s / self.T, dim=1)
        pred_teacher = F.softmax(y_t / self.T, dim=1)
        if self.weight is not None:
            pred_teacher = pred_teacher * self.weight
            pred_teacher = pred_teacher / pred_teacher.sum(1)[:, None]
        # s_mask_t = _get_gt_mask(y_s, target)

        s_mask_5 = top_mask(y_t, 5).int() - top_mask(y_t, 4).int()
        s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
        s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
        s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
        s_mask_1 = top_mask(y_t, 1).int()
        s_mask = top_mask(y_t, 5)
        not_s_mask = top_not_mask(y_t, 5)
        pred_student = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)
        pred_teacher = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, s_mask_5, not_s_mask)

        log_pred_student = torch.log(pred_student)
        loss_top7 = (
                F.kl_div(log_pred_student, pred_teacher, size_average=False)
                * (self.T ** 2)
                / target.shape[0]
        )

        pred_teacher_part2 = F.softmax(
            y_t / self.T - 1000 * s_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            y_s / self.T - 1000 * s_mask, dim=1
        )

        not_loss_top7 = F.kl_div(log_pred_student_part2, pred_teacher_part2,
                                 size_average=False) * (self.T ** 2) / target.size()[0]
        ce_loss = self.CELoss(y_s, target)
        kd = 1 * loss_top7 + 3 * not_loss_top7
        loss = alpha * (1 * loss_top7 + 3 * not_loss_top7) + ce_loss

        return loss, kd

def dkd_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, weight, logit_stand=True):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # if weight is not None:
    #     pred_teacher = pred_teacher * weight
    #     pred_teacher = pred_teacher / pred_teacher.sum(1)[:, None]

    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return ((alpha) * tckd_loss / 10 + (beta) * nckd_loss / 10) * 9


def cc_loss(logits_student, logits_teacher, target, alpha, beta, temperature, weight, reduce=True):
    #logits_student = normalize(logits_student)
    #logits_teacher = normalize(logits_teacher)

    batch_size, class_num = logits_teacher.shape
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # if weight is not None:
    #     pred_teacher = pred_teacher * weight
    #     pred_teacher = pred_teacher / pred_teacher.sum(1)[:, None]

    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)
    pred_student_o = F.softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_o = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    student_matrix = torch.mm(pred_student_t.transpose(1, 0), pred_student_t)
    teacher_matrix = torch.mm(pred_teacher_t.transpose(1, 0), pred_teacher_t)
    student_matrix1 = torch.mm(pred_student_o.transpose(1, 0), pred_student_o)
    teacher_matrix1 = torch.mm(pred_teacher_o.transpose(1, 0), pred_teacher_o)
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha/90 + (
                    (teacher_matrix1 - student_matrix1) ** 2).sum() * beta/90) / class_num /9
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha/90 + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta/90) / class_num /9
    return consistency_loss


def bc_loss(logits_student, logits_teacher, target, alpha, beta, temperature, weight, reduce=True):
    #logits_student = normalize(logits_student)
    #logits_teacher = normalize(logits_teacher)

    batch_size, class_num = logits_teacher.shape
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # if weight is not None:
    #     pred_teacher = pred_teacher * weight
    #     pred_teacher = pred_teacher / pred_teacher.sum(1)[:, None]

    pred_student_t = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher, gt_mask, other_mask)
    pred_student_o = F.softmax(logits_student / temperature - 1000 * gt_mask, dim=1)
    pred_teacher_o = F.softmax(logits_teacher / temperature - 1000 * gt_mask, dim=1)
    student_matrix = torch.mm(pred_student_t, pred_student_t.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher_t, pred_teacher_t.transpose(1, 0))
    student_matrix1 = torch.mm(pred_student_o, pred_student_o.transpose(1, 0))
    teacher_matrix1 = torch.mm(pred_teacher_o, pred_teacher_o.transpose(1, 0))
    if reduce:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2).sum() * alpha/90 + (
                (teacher_matrix1 - student_matrix1) ** 2).sum() * beta/90) / class_num/9
    else:
        consistency_loss = (((teacher_matrix - student_matrix) ** 2) * alpha/90 + (
                (teacher_matrix1 - student_matrix1) ** 2) * beta/90) / class_num/9
    return consistency_loss

class MDKDLoss(nn.Module):
    def __init__(self, cls_num_list, T, weight=None):
        super(MDKDLoss, self).__init__()
        self.temperature = T
        self.alpha = 1
        self.beta = 8
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()
        self.logit_stand = True

    #def forward_train(self, image_weak, image_strong, target, **kwargs):
    def forward(self, logits_student_weak, logits_student_strong, logits_teacher_weak, logits_teacher_strong, target, alpha):

        pred_teacher_weak = F.softmax(logits_teacher_weak.detach(), dim=1)
        confidence, pseudo_labels = pred_teacher_weak.max(dim=1)
        confidence = confidence.detach()
        conf_thresh = np.percentile(
            confidence.cpu().numpy().flatten(), 0
        )
        mask = confidence.le(conf_thresh).bool()

        class_confidence = torch.sum(pred_teacher_weak, dim=0)
        class_confidence = class_confidence.detach()
        class_confidence_thresh = np.percentile(
            class_confidence.cpu().numpy().flatten(), 0
        )
        class_conf_mask = class_confidence.le(class_confidence_thresh).bool()

        # losses
        loss_ce = self.CELoss(logits_student_weak, target) + self.CELoss(logits_student_strong, target)
        #loss_ce =  (F.cross_entropy(logits_student_weak, target) + F.cross_entropy(logits_student_strong, target))
        loss_kd_weak =  (alpha * ((dkd_loss(
        #loss_kd_weak=self.kd_loss_weight * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            weight=self.weight,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + alpha * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            3.0,
            weight=self.weight,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + alpha * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            5.0,
            weight=self.weight,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + alpha * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            weight=self.weight,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()) + alpha * ((dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            weight=self.weight,
            # reduce=False
            logit_stand=self.logit_stand,
        ) * mask).mean()))

        loss_kd_strong =  (alpha * dkd_loss(
        #loss_kd_strong = self.kd_loss_weight * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            weight=self.weight,
            logit_stand=self.logit_stand,
        ) + alpha * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            3.0,
            weight=self.weight,
            logit_stand=self.logit_stand,
        ) + alpha * dkd_loss(
            logits_student_strong,
            logits_teacher_strong,
            target,
            self.alpha,
            self.beta,
            5.0,
            weight=self.weight,
            logit_stand=self.logit_stand,
        ) + alpha * dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            weight=self.weight,
            logit_stand=self.logit_stand,
        ) + alpha * dkd_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            weight=self.weight,
            logit_stand=self.logit_stand,
        ))

        loss_cc_weak = (alpha * ((cc_loss(
        #loss_cc_weak= self.kd_loss_weight * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            weight=self.weight,
            # reduce=False
        ) * class_conf_mask).mean()) + alpha * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            3.0,
            weight=self.weight,
        ) * class_conf_mask).mean()) + alpha * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            5.0,
            weight=self.weight,
        ) * class_conf_mask).mean()) + alpha * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            weight=self.weight,
        ) * class_conf_mask).mean()) + alpha * ((cc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            weight=self.weight,
        ) * class_conf_mask).mean()))

        loss_bc_weak = (alpha * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            weight=self.weight,
        ) * mask).mean()) + alpha * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            3.0,
            weight=self.weight,
        ) * mask).mean()) + alpha * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            5.0,
            weight=self.weight,
        ) * mask).mean()) + alpha * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            2.0,
            weight=self.weight,
        ) * mask).mean()) + alpha * ((bc_loss(
            logits_student_weak,
            logits_teacher_weak,
            target,
            self.alpha,
            self.beta,
            6.0,
            weight=self.weight,
        ) * mask).mean()))
        kd_loss = loss_ce + loss_kd_weak + loss_kd_strong + loss_bc_weak + loss_cc_weak
        return  kd_loss

def top_7_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.ones_like(logits).scatter_(1, pred_s_index, 0).bool()
    return mask

def top_7_not_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.zeros_like(logits).scatter_(1, pred_s_index, 1).bool()
    return mask

def rekd_top_loss(y_s, y_t, target, temperature):

    #batch_size1 = y_t.size()[0]
    #simple_y_s = y_s.transpose(0, 1)
    #simple_y_t = y_t.transpose(0, 1)
    maxk = 7

    s_mask = top_7_mask(y_s, maxk)
    s_top7_logit = y_s / temperature - 1000 * s_mask
    t_top7_logit = y_t / temperature - 1000 * s_mask
    loss_top7_loss = F.kl_div(F.log_softmax(s_top7_logit, dim=1), F.softmax(t_top7_logit, dim=1), size_average=False) * (
                temperature * temperature) / target.size()[0]
    loss_top7 = F.kl_div(F.log_softmax(s_top7_logit, dim=1), F.softmax(t_top7_logit, dim=1), size_average=False,
                         reduction='batchmean').mean(dim=0) / target.size()[0]

    not_s_mask = top_7_not_mask(y_s, maxk)

    s_not_top7_logit = y_s / temperature - 1000 * not_s_mask
    t_not_top7_logit = y_t / temperature - 1000 * not_s_mask

    not_loss_top7_loss = F.kl_div(F.log_softmax(s_not_top7_logit, dim=1), F.softmax(t_not_top7_logit, dim=1),
                             size_average=False) * (temperature * temperature) / target.size()[0]
    not_loss_top7 = F.kl_div(F.log_softmax(s_not_top7_logit, dim=1), F.softmax(t_not_top7_logit, dim=1),
                                  size_average=False, reduction='batchmean').mean(dim=0) / target.size()[0]
    return loss_top7_loss + not_loss_top7_loss, loss_top7 + not_loss_top7

class HKDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T, weight=None):
        super(HKDLoss, self).__init__()
        self.T = T
        # self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, out_s, out_t, target, alpha):
        kd_loss, kd = rekd_top_loss(out_s, out_t, target, self.T)
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd


def cat_mask(t, mask1, mask2, mask3, mask4, not_mask):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    t3 = (t * mask3).sum(dim=1, keepdims=True)
    t4 = (t * mask4).sum(dim=1, keepdims=True)
    tn = (t * not_mask).sum(1, keepdims=True)
    rt = torch.cat([t1, t2, t3, t4, tn], dim=1)
    return rt

def top_not_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.ones_like(logits).scatter_(1, pred_s_index, 0).bool()
    return mask

def top_mask(logits, maxk):
    pred_s_value, pred_s_index = logits.topk(maxk, 1, True, True)
    mask = torch.zeros_like(logits).scatter_(1, pred_s_index, 1).bool()
    return mask

def hc_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, weight, logit_stand=False):
    y_s = normalize(logits_student_in) if logit_stand else logits_student_in
    y_t = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    pred_student = F.softmax(y_s / temperature, dim=1)
    pred_teacher = F.softmax(y_t / temperature, dim=1)
    #s_mask_t = _get_gt_mask(y_s, target)
    if weight is not None:
        pred_teacher = pred_teacher * weight
        pred_teacher = pred_teacher / pred_teacher.sum(1)[:, None]

    s_mask_4 = top_mask(y_t, 4).int() - top_mask(y_t, 3).int()
    s_mask_3 = top_mask(y_t, 3).int() - top_mask(y_t, 2).int()
    s_mask_2 = top_mask(y_t, 2).int() - top_mask(y_t, 1).int()
    s_mask_1 = top_mask(y_t,1).int()
    s_mask = top_mask(y_t, 4)
    not_s_mask = top_not_mask(y_t, 4)
    pred_student = cat_mask(pred_student, s_mask_1, s_mask_2, s_mask_3, s_mask_4, not_s_mask)
    pred_teacher = cat_mask(pred_teacher, s_mask_1, s_mask_2, s_mask_3, s_mask_4, not_s_mask)

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

    return alpha * loss_top7 + beta * not_loss_top7, alpha * loss_top7.mean(dim=0) + beta * not_loss_top7.mean(dim=0)

class HDLDLoss(nn.Module):


    def __init__(self, cls_num_list, T, weight=None):
        super(HDLDLoss, self).__init__()
        self.T = T
        #self.T = torch.cuda.FloatTensor([1, 2, 3, 4, 5, 6, 7, 4.5, 5, 5.5])
        self.weight = weight
        self.class_freq = torch.cuda.FloatTensor(cls_num_list / np.sum(cls_num_list))
        self.CELoss = nn.CrossEntropyLoss(weight=self.weight).cuda()

    def forward(self, out_s, out_t, target, alpha):
        kd_loss, kd = hc_loss(out_s, out_t, target, 0.5, 1, self.T, weight=self.weight)
        ce_loss = self.CELoss(out_s, target)
        loss = alpha * kd_loss + ce_loss

        return loss, kd