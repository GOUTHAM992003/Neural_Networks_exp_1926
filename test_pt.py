import torch
import torch.nn as nn
linear = nn.Linear(768, 768).cuda()
x = torch.randn(8, 1024, 768).cuda()
y = linear(x)
torch.cuda.synchronize()
