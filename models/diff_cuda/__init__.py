import torch
import torch.nn.functional as F


class Diff_cuda(torch.nn.Module):
    def __init__(self):
        super(Diff_cuda, self).__init__()
        kernel = torch.ones((3, 3))
        kernel = torch.FloatTensor(kernel).expand(64, 1, 3, 3)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x1, x2):
        x = torch.pow(x1 - x2, 2)
        weight = self.weight.cuda()
        x = F.conv2d(x, weight, stride=1, padding=1, groups=64)
        return torch.tanh(x)
