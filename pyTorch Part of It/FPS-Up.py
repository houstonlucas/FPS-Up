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

    dtype = torch.FloatTensor

    img1 = frames[0]
    img2 = frames[2]

    # Format the images to be a 2xhxw ndarray
    imgs = np.stack((img1, img2), 0)

    # Convert the images into a torch Tensor
    inputs = Variable(torch.from_numpy(imgs).type(dtype))

    # Push the images through the network
    outputs = fup(inputs)
    print("Output size: " + str(outputs.size()))
    img_out = outputs.data.numpy()
    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.imshow("Output", img_out.astype(np.uint8))
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
        img_out = img_out.squeeze(0).permute(1,2,0)
        print("img_out size (inNet)", img_out.size())

        return img_out


def get_video_frames(file_name):
    cap = cv2.VideoCapture(file_name)

    frames = []

    r, frame = cap.read()
    while r:
        frames.append(frame)
        r, frame = cap.read()

    return frames


if __name__ == '__main__':
    main()
