import torch
import torch.nn as nn

def conv3x3(inp, oup, stride=1, groups=1, dilation=1):
    return nn.Conv2d(inp, oup, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


inputs = torch.randn(1, 1, 32, 32)
conv1 = conv3x3(1, 1, stride=2, groups=1)
conv2 = conv3x3(1, 1, stride=1, groups=1)

h1 = conv1(inputs)
h2 = conv2(h1)

print(h1.shape)
print(h2.shape)