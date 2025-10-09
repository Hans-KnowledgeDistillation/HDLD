import torch
import torch.nn as nn
import torchvision.models as models
from mdistiller.models.cifar.resnet import resnet8x4,resnet32x4
#from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format

net = resnet32x4()
net.eval()
input = torch.randn(1, 3, 32, 32)

MACs, params = profile(net, inputs=(input,))

MACs, params = clever_format([MACs, params], '%.3f')
print(f"{MACs},{params}")
# with torch.cuda.device(0):
#     net = models.resnet8x4()
#     flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True,
#                                               verbose=True)
#     print('Flops:  ', flops)
#     print('Params: ', params)