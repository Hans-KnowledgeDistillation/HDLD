import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils_fgsm import load_checkpoint, validate, validate_FGSM
from mdistiller.engine.cfg import CFG as cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])
    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    eps = 0.001
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.002
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.003
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.004
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.005
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.006
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.007
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.008
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.009
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.01
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.011
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.012
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.013
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.014
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.015
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.016
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.017
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.018
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.019
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)
    eps = 0.02
    test_acc, test_acc_top5, test_loss = validate_FGSM(val_loader, model, eps)