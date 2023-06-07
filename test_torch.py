
import torch.nn as nn
import torch
import numpy as np
from unet import UNet

################ CrossEntropyLoss how to do
def test_CrossEntropyLoss():
    num = 4
    x = torch.rand((1, 2, num, num))
    y = torch.rand((1, num, num))
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    y = y.long()
    print(x)
    print(y)
    softmax = nn.Softmax(dim=1)
    x_softmax = softmax(x)
    print("x_softmax=", x_softmax)
    x_log = torch.log(x_softmax)
    print("x_log=", x_log)
    x_permute = x_log.permute(0, 2, 3, 1)
    print("x_permute=", x_permute)
    print(x_permute.shape)
    loss = 0.0
    for i in np.arange(len(x_permute[0])):
        for j in np.arange(len(x_permute[0][0])):
            loss += abs(x_permute[0][i][j][0]) if y[0][i][j] == 0 else abs(x_permute[0][i][j][1])

    print("loss=", loss, "  ", loss * 1.0 / (num * num))  # mean
    crossEntropyLoss = nn.CrossEntropyLoss()
    loss_1 = crossEntropyLoss(x, y)
    print("loss_1=", loss_1)


test_CrossEntropyLoss()

# model1 = UNet(1, 2)
# for n in model1.modules():
#     print(n)
