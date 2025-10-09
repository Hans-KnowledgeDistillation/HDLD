import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn
import sys, os
sys.path.append(os.getcwd())
print
sys.path.append("/home/lv/yzh/2025/logit-standardization-KD-master/")

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


def get_output_metric_oneindex(model, val_loader, num_classes=100, ifstand=False):
    model.eval()
    index = 7  # 7, 12
    all_preds, all_labels = None, None
    sample = None
    label = None
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            if i != index:
                continue
            sample = data[0]
            label = labels[0]
            #             plt.imshow(sample.permute(1,2,0))
            outputs, _ = model(data)
            if ifstand:
                preds = normalize(outputs)
            else:
                preds = outputs
            all_preds = preds.data.cpu().numpy()
            all_labels = labels.data.cpu().numpy()
    print(all_preds.shape, label)
    print(all_preds.mean(-1).max(), all_preds.std(-1).max())
    return all_preds[0], all_preds.std(-1).max()


def get_tea_stu_diff_oneindex(tea, stu, mpath, ifstand=False):
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
    ms, maxstd_s = get_output_metric_oneindex(model, val_loader, ifstand=ifstand)
    mt, maxstd_t = get_output_metric_oneindex(tea_model, val_loader, ifstand=ifstand)
    font_size = 18
    x = list(range(100))  # ["1"]*100
    xlabels = []
    for ii in range(10):
        xlabels += [ii * 10]
        xlabels += [''] * 9
    plt.figure(figsize=(9, 5))
    ax = seaborn.barplot(x=np.array(x), y=mt, color='blue')
    ax = seaborn.barplot(x=np.array(x), y=ms, color='red')
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_xticklabels(xlabels, fontsize=17)
    topbar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc='blue', edgecolor='none')
    l = plt.legend([bottombar, topbar], ['teacher, std.=%.2f' % (maxstd_t), 'student, std.=%.2f' % (maxstd_s)], loc=1,
                   ncol=1, prop={'size': font_size})
    l.draw_frame(False)
    plt.yticks(fontsize=17)
    plt.ylim(-5, 12.5)
    plt.tight_layout()
    plt.xlabel("class category", fontsize=font_size)
    plt.ylabel("logit value", fontsize=font_size)
    plt.show()
    plt.close()

    #     plt.figure(figsize=(7,5))
    #     ax.set_xticklabels(xlabels, fontsize=17)
    #     plt.yticks(fontsize=17)
    #     plt.xlabel("class", fontsize = 20)
    #     plt.ylabel("logit", fontsize = 20)
    #     plt.show()
    print(ms.mean(), mt.mean())
    return ms, mt

def get_stu_diff_oneindex(stu, mpath, ifstand=False):
    cfg.defrost()
    cfg.DISTILLER.STUDENT = stu
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    model = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
    model.load_state_dict(load_checkpoint(mpath)["model"])

    print("load model successfully!")
    ms, maxstd_s = get_output_metric_oneindex(model, val_loader, ifstand=ifstand)

    font_size = 18
    x = list(range(100))  # ["1"]*100
    xlabels = []
    for ii in range(10):
        xlabels += [ii * 10]
        xlabels += [''] * 9
    plt.figure(figsize=(8, 4))

    ax = seaborn.barplot(x=np.array(x), y=ms, color='red')
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_xticklabels(xlabels, fontsize=17)
    topbar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc='red', edgecolor='none')
    l = plt.legend([bottombar, topbar], ['wrn_40_1'], loc=1,
                   ncol=1, prop={'size': font_size})
    l.draw_frame(False)
    plt.yticks(fontsize=17)
    plt.ylim(-8.5, 14)
    plt.tight_layout()
    plt.xlabel("class category", fontsize=font_size)
    plt.ylabel("logit value", fontsize=font_size)
    plt.show()
    plt.close()

    #     plt.figure(figsize=(7,5))
    #     ax.set_xticklabels(xlabels, fontsize=17)
    #     plt.yticks(fontsize=17)
    #     plt.xlabel("class", fontsize = 20)
    #     plt.ylabel("logit", fontsize = 20)
    #     plt.show()
    print(ms.mean())
    return ms

def get_tea_diff_oneindex(tea1, ifstand=False):
    cfg.defrost()
    cfg.DISTILLER.TEACHER = tea1
    cfg.DATASET.TYPE = 'cifar100'
    cfg.freeze()
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    tea_model1 = cifar_model_dict[cfg.DISTILLER.TEACHER][0](num_classes=num_classes)
    tea_model1.load_state_dict(load_checkpoint(cifar_model_dict[cfg.DISTILLER.TEACHER][1])["model"])

    print("load model successfully!")
    mt1, maxstd_t1 = get_output_metric_oneindex(tea_model1, val_loader, ifstand=ifstand)

    font_size = 18
    x = list(range(100))  # ["1"]*100
    xlabels = []
    for ii in range(10):
        xlabels += [ii * 10]
        xlabels += [''] * 9
    plt.figure(figsize=(9, 5))
    ax = seaborn.barplot(x=np.array(x), y=mt1, color='blue')

    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.set_xticklabels(xlabels, fontsize=17)
    topbar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='none')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc='blue', edgecolor='none')
    l = plt.legend([bottombar, topbar], ['wrn_40_2'], loc=1,
                   ncol=1, prop={'size': font_size})
    l.draw_frame(False)
    plt.yticks(fontsize=17)
    plt.ylim(-5, 12)
    plt.tight_layout()
    plt.xlabel("class category", fontsize=font_size)
    plt.ylabel("logit value", fontsize=font_size)
    plt.show()
    plt.close()

    #     plt.figure(figsize=(7,5))
    #     ax.set_xticklabels(xlabels, fontsize=17)
    #     plt.yticks(fontsize=17)
    #     plt.xlabel("class", fontsize = 20)
    #     plt.ylabel("logit", fontsize = 20)
    #     plt.show()
    print(mt1.mean())
    return mt1

mpath = "/home/lv/yzh/2025/logit-standardization-KD-master/output/cifar100_baselines_student/wrn_40_1/student_latest"
#mt1, mt2 = get_tea1_tea2_diff_oneindex("vgg13", "resnet32x4")
mt1 = get_tea_diff_oneindex("wrn_40_2")
#ms = get_stu_diff_oneindex("wrn_40_1", mpath)