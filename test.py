#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/7/29 15:55
# @Author : fhh
# @FileName: test.py
# @Software: PyCharm

import torch
torch.manual_seed(20230328)

a = torch.randn(2, 4, 3)
print(a)

b, _ = torch.max(a, dim=2, keepdim=True)
print(b)
print(b.shape)


c, _ = torch.max(a, dim=2)
print(c)
print(c.shape)

m = torch.nn.AdaptiveMaxPool1d(1)
input = torch.randn(1, 64, 8)
output = m(input)
print(output.shape)

for j, k in enumerate(a, input):
    print(j)
    print(k)
