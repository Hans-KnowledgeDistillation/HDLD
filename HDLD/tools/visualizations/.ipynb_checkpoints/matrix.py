import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import sys, os
sys.path.append(os.path.join(os.getcwd(),'../..'))
print(os.getcwd())

from mdistiller.models import cifar_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.cfg import CFG as cfg


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / stdv


def get_output_metric(model, val_loader, num_classes=100, ifstand=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            outputs, _ = model(data)
            if ifstand:
                preds = normalize(outputs)
            else:
                preds = outputs
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)
    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))
    for p, l in zip(all_preds, all_labels):
        cnt[l, 0] += 1
        matrix[l] += p
    matrix /= cnt
    return matrix


def get_tea_stu_diff(tea, stu, mpath, max_diff, ifstand=False):
    cfg.defrost()
    cfg.DISTILLER.STUDENT = stu
    cfg.DISTILLER.TEACHER = tea
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    model.load_state_dict(load_checkpoint(mpath)["model"])
    tea_model = cifar_model_dict[cfg.DISTILLER.TEACHER][0](num_classes=num_classes)
    tea_model.load_state_dict(load_checkpoint(cifar_model_dict[cfg.DISTILLER.TEACHER][1])["model"])
    print("load model successfully!")
    ms = get_output_metric(model, val_loader, ifstand=ifstand)
    mt = get_output_metric(tea_model, val_loader, ifstand=ifstand)
    diff = np.abs((ms - mt))
    for i in range(100):
        diff[i, i] = 0
    print('max(diff):', diff.max(), diff.argmax())
    print('mean(diff):', diff.mean())
    #     diff[diff>0.1] /= 10
    seaborn.heatmap(diff, vmin=0, vmax=max_diff, cmap="Reds", cbar_kws=dict(use_gridspec=False, location="left"))
    #     plt.tight_layout()
    plt.show()
    return diff

MAX_DIFF = 3.

mpath = "/home/lv/yzh/2025/logit-standardization-KD-master/output/comp/dkd,res32x4,res8x4/student_best"
diff = get_tea_stu_diff("resnet32x4", "resnet8x4", mpath, MAX_DIFF)