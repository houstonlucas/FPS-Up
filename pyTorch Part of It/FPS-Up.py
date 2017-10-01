from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt


def main():
    pass


class FPS_UP(nn.Module):
    def __init__(self):
        super(FPS_UP, self).__init__()

        self.k = 6
        self.d = 5

        self.conv1 = nn.Conv2d(1, self.k, self.d)
        self.conv2 = nn.Conv2d(1, self.k, self.d)

        # TODO learn about padding
        self.deConv = nn.ConvTranspose2d(2 * self.k, 1, self.d)

    def forward(self, img1, img2):
        c1 = self.conv1(img1)
        c2 = self.conv2(img2)
        a = torch.stack((c1, c2), 0)
        imgOut = self.deConv(a)
        return imgOut


if __name__ == '__main__':
    main()
