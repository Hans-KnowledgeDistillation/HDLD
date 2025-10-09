import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg

###########################
def fgsm_attack(image, data_grad, epsilon):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon * sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    #perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
    # 返回对抗样本
    return perturbed_image

def validate_FGSM(val_loader, distiller, epsilon):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    start_time = time.time()
    for idx, (image, target) in enumerate(val_loader):
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        image.requires_grad_(True)
        output_original = distiller(image=image)
        loss_originnal = criterion(output_original, target)

        distiller.zero_grad()
        loss_originnal.backward()
        data_grad = image.grad.data

        perturbed_image = fgsm_attack(image, data_grad, epsilon)
        output_adv = distiller(image=perturbed_image)
        loss_adv = criterion(output_adv, target)

        acc1, acc5 = accuracy(output_adv, target, topk=(1, 5))
        batch_size = perturbed_image.size(0)
        losses.update(loss_adv.cpu().detach().numpy().mean(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
        pbar.set_description(log_msg(msg, "EVAL"))
        pbar.update()

    pbar.close()
    return top1.avg, top5.avg, losses.avg

def pgd_attack(model, image, criterion, target, epsilon, alpha, iters):
    perturbed_images = image.clone().detach()
    #if random_start:
    perturbed_images = perturbed_images + torch.empty_like(image).uniform_(-epsilon, epsilon)
        #perturbed_images = torch.clamp(perturbed_images, 0, 1)

    for _ in range(iters):
        perturbed_images.requires_grad_(True)

        output_adv = model(image=perturbed_images)
        loss_adv = criterion(output_adv, target)
        model.zero_grad()
        loss_adv.backward()

        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        # 通过epsilon生成对抗样本
        perturbed_images = perturbed_images + alpha * epsilon * sign_data_grad


    #perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

def validate_PGD(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    start_time = time.time()
    for idx, (image, target) in enumerate(val_loader):
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        image.requires_grad_(True)
        output_original = distiller(image=image)
        loss_originnal = criterion(output_original, target)

        distiller.zero_grad()
        loss_originnal.backward()
        #data_grad = image.grad.data
        perturbed_image = pgd_attack(model=distiller, image=image, criterion=criterion, target=target,
                                     epsilon=0.001, alpha=0.1, iters=10)

        output_adv = distiller(image=perturbed_image)
        loss_adv = criterion(output_adv, target)

        acc1, acc5 = accuracy(output_adv, target, topk=(1, 5))
        batch_size = perturbed_image.size(0)
        losses.update(loss_adv.cpu().detach().numpy().mean(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
        pbar.set_description(log_msg(msg, "EVAL"))
        pbar.update()

    pbar.close()
    return top1.avg, top5.avg, losses.avg

def validate_npy(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        start_eval = True
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            output = nn.Softmax()(output)
            if start_eval:
                all_image = image.float().cpu()
                all_output = output.float().cpu()
                all_label = target.float().cpu()
                start_eval = False
            else:
                all_image = torch.cat((all_image, image.float().cpu()), dim=0)
                all_output = torch.cat((all_output, output.float().cpu()), dim=0)
                all_label = torch.cat((all_label, target.float().cpu()), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    all_image, all_output, all_label = all_image.numpy(), all_output.numpy(), all_label.numpy()
    pbar.close()
    return top1.avg, top5.avg, losses.avg, all_image, all_output, all_label


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
