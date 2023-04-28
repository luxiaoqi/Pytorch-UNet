import argparse
import numpy
#import dicom
import pydicom as dicom
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def getFiles(path, ext=''):
    filesTemp = []
    for root, dirs, files in os.walk(path):
        for file in files:
            res=os.path.splitext(file)
            if len(res) != 0 and res[1] != ext:
                continue
            filePath = os.path.join(root, file)
            filesTemp.append(filePath)
    return filesTemp


def dcm2bmp(resource, des):
    res=os.path.splitext(resource)
    if len(res) != 2 or res[1] != '.dcm':
        return
    if os.path.exists(resource) == False:
        return
    res = os.path.splitext(des)
    if len(res) != 2:
        return
    readDCM = dicom.read_file(resource)
    image = np.array(readDCM.pixel_array)
    RescaleSlope = readDCM.RescaleSlope
    RescaleIntercept = readDCM.RescaleIntercept
    WindowCenter = readDCM.WindowCenter
    WindowWidth = readDCM.WindowWidth
    max_val = np.max(image)
    min_val = np.min(image)
    img_arr = (image - min_val)*255 / (max_val - min_val)
    img_arr = img_arr.astype(np.uint8)

    #show image view
    # cv2.imshow("image show",img_arr)
    # cv2.waitKey(0)

    cv2.imwrite(des, img_arr)

    #show image view
    # img_arr=cv2.resize(img_arr, (256, 256), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("image show",img_arr)
    # cv2.waitKey(0)

def get_args():
    parser = argparse.ArgumentParser(description='convert dcm to bmp for labelme')
    parser.add_argument('inDir')
    return parser.parse_args()

def main():
    args = get_args()
    path = args.inDir #'D:\workspace\Scout\scout_cardiac'
    files = getFiles(path, '.dcm')
    print(len(files))
    #print(files)
    for file in files:
        res = os.path.splitext(file)
        bmpfile = res[0] + '.bmp'
        dcm2bmp(file, bmpfile)

        # temp = cv2.imread(bmpfile, flags=0)
        # print(temp.shape)

def showImage():
    path='D:\\workspace\\Pytorch-UNet\\data\\test\\776_101_1480_OUT.png'
    img_arr = cv2.imread(path)
    mmax = np.max(img_arr)
    mmin = np.min(img_arr)
    cv2.imshow("image show", img_arr)
    cv2.waitKey(0)
    # img = Image.open(path)
    # ar = np.array(img)
    # img.show()


if __name__== "__main__" :
    main()