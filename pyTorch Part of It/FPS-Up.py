from __future__ import print_function
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import matplotlib.pyplot as plt


def main():
    frames = get_video_frames("../small.mp4")
    h, w, _ = frames[0].shape
    fup = FPS_UP(h, w)

    generator = training_data_generator(frames)

    inputs, expected_out = next(generator)

    target = to_numpy(expected_out)

    # Push the images through the network
    output = fup(inputs)

    img_out = to_numpy(output)

    cv2.imshow("Target", target)
    cv2.imshow("Output", img_out)
    cv2.waitKey(-1)


class FPS_UP(nn.Module):
    def __init__(self, h, w):
        super(FPS_UP, self).__init__()

        self.img_height = h
        self.img_width = w

        self.k = 6
        self.d = 5

        self.conv1 = nn.Conv2d(3, self.k, self.d)
        self.conv2 = nn.Conv2d(3, self.k, self.d)

        # TODO learn about padding
        self.deConv = nn.ConvTranspose2d(2 * self.k, 3, self.d)

    def forward(self, imgs):
        img1 = imgs[0].permute(2, 0, 1).unsqueeze(0)
        print(img1.size())
        img2 = imgs[1].permute(2, 0, 1).unsqueeze(0)
        c1 = self.conv1(img1)
        c2 = self.conv2(img2)
        a = torch.cat((c1, c2), 1)
        img_out = self.deConv(a)
        img_out = img_out.squeeze(0).permute(1, 2, 0)
        return img_out


def get_video_frames(file_name):
    cap = cv2.VideoCapture(file_name)

    frames = []

    r, frame = cap.read()
    while r:
        frames.append(frame)
        r, frame = cap.read()

    return frames


# Generator for producing training examples
def training_data_generator(frames):
    n = len(frames)
    for i in range(n - 3):
        A, B, C = frames[i:i + 3]
        numpy_pair = np.stack((A, C), 0)
        torch_pair = to_torch_variable(numpy_pair)
        target = to_torch_variable(B)
        yield torch_pair, target


# Converts a numpy ndarray to a torch variable
def to_torch_variable(nump, torchType=torch.FloatTensor):
    return Variable(torch.from_numpy(nump).type(torchType))


# Converts pytorch vars to the numpy format wanted.
def to_numpy(var, dtype=np.uint8):
    return var.data.numpy().astype(dtype)


if __name__ == '__main__':
    main()
