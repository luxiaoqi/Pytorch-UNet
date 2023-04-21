from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import math

imageDir = "data\imgs"
maskDir = "data\masks"
def getFileVec(dir):
    fileVec = []
    for root, dirs, files in os.walk(dir):
        if root != dir:
            break
        for file in files:
            path = os.path.join(root, file)
            fileVec.append(path)
    return fileVec

fileVec = getFileVec(imageDir)
print(len(fileVec))


def drawImage():
    for i in range(80):
        index = i + 1
        imageFile = '/PNGImages/FudanPed%.5d.png' % index
        markFile = '/PedMasks/FudanPed%.5d_mask.png' % index
        image = Image.open(imageDir + imageFile)
        mask = Image.open(imageDir + markFile)
        plt.subplot(2, 2, 1)
        plt.imshow(mask)
        mask.putpalette([
            0, 0, 0,  # black background
            255, 0, 0,  # index 1 is red
            255, 255, 0,  # index 2 is yellow
            255, 153, 0,  # index 3 is orange
        ])
        plt.subplot(2, 2, 4)
        plt.imshow(mask)
        plt.show()



