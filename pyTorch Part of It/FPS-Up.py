from __future__ import print_function
import torch
from torch.autograd import Variable
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from PIL import Image

import cv2

import matplotlib.pyplot as plt

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

def main():
    frames = get_video_frames("../small.mp4")
    h, w = frames[0].size
    fup = FPS_UP(h, w)

    dtype = torch.FloatTensor

    img1 = frames[0]
    img2 = frames[2]
    #print(img1)

    #print(imgs)

    inp1 = Variable((preprocess(img1)).unsqueeze_(0))
    # inp1.byte()
    inp2 = Variable((preprocess(img2)).unsqueeze_(0))
    # inp2.byte()

    inputs = (inp1, inp2)
    print(inputs.size())

    outputs = fup.forward(inps)
    print(outputs.size())

    img_out = outputs.data.numpy()[0].reshape(h, w, 3)
    print(img_out.shape, img1.shape)
    cv2.imshow("Output", img_out)
    cv2.imshow("Original", img1)
    cv2.imshow("Original - Output", inp1.data.numpy())
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
        self.deConv = nn.ConvTranspose2d(self.k, 3, self.d)

    def forward(self, imgs):
        img1 = imgs[0].view(1, 3, self.img_height, self.img_width)
        img2 = imgs[1].view(1, 3, self.img_height, self.img_width)
        c1 = self.conv1(img1)
        c2 = self.conv2(img2)
        a = torch.cat((c1, c2), 0)
        img_out = self.deConv(a)
        return img_out


def get_video_frames(file_name):
    cap = cv2.VideoCapture(file_name)

    frames = []

    r, frame = cap.read()
    while r:
        temp_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # img_tensor = preprocess(temp_image)
        # img_tensor.unsqueeze_(0)
        # img_variable = Variable(img_tensor)
        frames.append(temp_image)
        r, frame = cap.read()

    return frames


if __name__ == '__main__':
    main()
