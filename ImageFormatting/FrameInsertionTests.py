import cv2
import numpy as np

def main():
    frames = getVideoFrames("../small.mp4")
    print(len(frames))

def getVideoFrames(file_name):
    cap = cv2.VideoCapture(file_name)

    frames = []

    r, frame = cap.read()
    while r:
        frames.append(frame)
        r, frame = cap.read()

    return frames


if __name__ == "__main__":
    main()
