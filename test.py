import torch
from torchvision import models
from pretrainedmodels.models.senet import se_resnext50_32x4d
import pretrainedmodels

import torch.nn as nn

# net = nn.Sequential(
#     nn.Linear(4,2),
#     nn.Sigmoid()
# )

# a = torch.randn(4,4)
# print(a)
# print(net(a))
# print(*list(models.resnext50_32x4d(pretrained=None).children())[:-2])
# print(torch.cuda.device_count())
import os
f = open(os.path.join('./', 'log.txt'), 'a')
criterion = nn.CrossEntropyLoss()
output = torch.tensor([[1,2,3,4]])
labels = torch.tensor([[2,3,8,5]])
# loss = criterion(output, labels)
print(torch.max( labels, 1))