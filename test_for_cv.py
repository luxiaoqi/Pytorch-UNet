# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np
import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

windowname = "lxq"

def cv2_do():
    # cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)  # 在window_normal模式下更改窗口大小
    # cv2.resizeWindow(windowname, 800, 600)  # 窗口更改为800x600大小的尺寸
    # key = cv2.waitKey(3000)
    # cv2.destroyWindow(windowname)

    # img_arr = cv2.imread('image\\11_101_29.png', 0)
    # img_arr = np.zeros((255, 255, 3),dtype=np.uint8)
    # cv2.imshow('one', img_arr)
    # img_arr[0:20, 3:20, 2] = 255
    # #print(img_arr, '\n', img_arr.item(3, 6, 2))
    # cv2.imshow('two', img_arr)
    # cv2.waitKey(20000)
    # cv2.destroyAllWindows()

    # plt.imshow(img_arr, cmap=plt.cm.gray)
    # plt.show()

    # lena = cv2.imread(r'image\\11_101_29.png', -1)
    # dollar = cv2.imread(r'image\\56_101_110.png', -1)
    # cv2.imshow('lena', lena)
    # cv2.imshow('dollar', dollar)
    # lena_face = lena[220:400, 250:350]
    # dollar[160:340, 200:300] = lena_face
    # cv2.imshow('dollar2', dollar)
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()

    # a = np.random.randint(0, 256, (5, 5), dtype=np.uint8)  # 原图像-灰度图像
    # mask = np.zeros((5, 5), dtype=np.uint8)  # 掩码图像
    # mask[0:3, 0:3] = 255
    # mask[4, 4] = 255
    # b = cv2.bitwise_or(a, mask)  # 原图像与掩码图像的按位与操作。
    # print(a.shape)

    # img1 = np.ones((4, 4), dtype=np.uint8) * 3
    # img2 = np.ones((4, 4), dtype=np.uint8) * 5
    # mask = np.zeros((4, 4), dtype=np.uint8)
    # mask[2:4, 2:4] = 1
    # img3 = cv2.add(img1, img2, mask=mask)
    # img1, img2
    # mask, img3

    # colorlena = cv2.imread(r'image\\11_101_29.png')
    # w, h, c = colorlena.shape
    # mask = np.zeros((w, h), dtype=np.uint8)
    # mask[0:50, 2:280] = 1
    # mask[100:500, 100:200] = 1
    # #result = cv2.bitwise_and(colorlena, colorlena, mask=mask)
    # result = cv2.bitwise_and(colorlena, colorlena, mask=mask)
    # cv2.imshow('colorlena', colorlena)
    # cv2.imshow('mask', mask)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # lena = cv2.imread(r'image\\11_101_29.png', 0)
    # cv2.imshow('lena', lena)
    # r, c = lena.shape
    # list_img = []  # 生成一个空列表，保存用循环生成的每个位图
    # for i in range(8):
    #     mat = np.ones((r, c), dtype=np.uint8) * 2 ** i  # 生成提取矩阵mat
    #     bit_img = cv2.bitwise_and(lena, mat)  # 原始图像与提取矩阵进行按位与运算，得到对应位上的数据。！！！注意
    #     bit_img[bit_img[:, :] != 0] = 255
    #     list_img.append(bit_img)
    # cv2.imshow('lena0', list_img[0])
    # cv2.imshow('lena1', list_img[1])
    # cv2.imshow('lena2', list_img[2])
    # cv2.imshow('lena3', list_img[3])
    # cv2.imshow('lena4', list_img[4])
    # cv2.imshow('lena5', list_img[5])
    # cv2.imshow('lena6', list_img[6])
    # cv2.imshow('lena7', list_img[7])
    # cv2.waitKey(30000)
    # cv2.destroyAllWindows()

    # # 例4.6 测试RGB色彩空间中不同颜色的值转化到HSV色彩空间后的对应值：
    # img = np.zeros((200, 200, 3), dtype=np.uint8)
    # # bgr图片中的蓝色映射到hsv中：
    # img_blue = img.copy()
    # img_blue[:, :, 0] = 255  # 把img的b通道值置为255，此时图像就是一张蓝色图像。
    # print('bgr图片中的蓝色数据=\n', img_blue[:3, :3, :])  # 把b的前3行前3列切出来展示
    # img_blue_hsv = cv2.cvtColor(img_blue, cv2.COLOR_BGR2HSV)
    # print('\nbgr中的蓝色映射到hsv中的数据=\n', img_blue_hsv[:3, :3, :])  # 把转化位HSV的b的前3行前3列切出来展示
    # # bgr图片中的绿色映射到hsv中：
    # img_green = img.copy()
    # img_green[:, :, 1] = 255
    # print('\nbgr图片中的绿色数据=\n', img_green[:3, :3, :])
    # img_green_hsv = cv2.cvtColor(img_green, cv2.COLOR_BGR2HSV)
    # print('\nbgr中的绿色映射到hsv中的数据=\n', img_green_hsv[:3, :3, :])
    # # bgr图片中的红色映射到hsv中：
    # img_red = img.copy()
    # img_red[:, :, 2] = 255
    # print('\nbgr图片中的红色数据=\n', img_red[:3, :3, :])
    # img_red_hsv = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
    # print('\nbgr中的红色映射到hsv中的数据=\n', img_red_hsv[:3, :3, :])
    # cv2.imshow('img_blue', img_blue)
    # cv2.imshow('img_blue_hsv', img_blue_hsv)
    # cv2.imshow('img_green', img_green)
    # cv2.imshow('img_green_hsv', img_green_hsv)
    # cv2.imshow('img_red', img_red)
    # cv2.imshow('img_red_hsv', img_red_hsv)
    # cv2.waitKey(50000)
    # cv2.destroyAllWindows()

    # # 例4.7 练习cv2.inRange()函数，将图像内[100,200]内的像素点提取出来
    # img = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    # mask = cv2.inRange(img, 100, 200)  # 制作掩码图像
    # img_show = cv2.bitwise_and(img, img, mask=mask)  # 按位与运算提取特定像素点
    # print(img_show)

    # # 例4.11 调整hsv图像的v值，实现艺术效果
    # img = cv2.imread(r'image\\lenacolor.png')
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(img_hsv)
    # #v[:, :] = 255
    # #s[:, :] = 255
    # img_new_hsv = cv2.merge([h, s, v])
    # img_new = cv2.cvtColor(img_new_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('img', img)
    # cv2.imshow('img_new', img_new)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


    # lenacolor = cv2.imread(r'image\\lenacolor.png')
    # lenacolor_bgra = cv2.cvtColor(lenacolor, cv2.COLOR_BGR2BGRA)
    # b, g, r, a = cv2.split(lenacolor_bgra)
    # lenacolor_bgra = cv2.merge([r, g, b, a])
    # a[:, :] = 100
    # lenacolor_bgra_new = cv2.merge([r, g, b, a])
    # a[:, :] = 20
    # lenacolor_bgra_new1 = cv2.merge([r, g, b, a])
    # temp = lenacolor[:, :, ::-1]
    # plt.subplot(141), plt.imshow(lenacolor[:, :, ::-1])
    # plt.subplot(142), plt.imshow(lenacolor_bgra)
    # plt.subplot(143), plt.imshow(lenacolor_bgra_new)
    # plt.subplot(144), plt.imshow(lenacolor_bgra_new1)
    # plt.show()

    # img = np.random.randint(0, 256, (4, 5), np.uint8)
    # mapx = np.ones((3, 3), np.float32)*2
    # mapy = np.ones((3, 3), np.float32) * 2.4
    # result = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # result

    #均值滤波
    img = cv2.imread('image\\lenacolor.png')
    img_blur1=cv2.blur(img, (3,3))
    img_blur2=cv2.blur(img, (30,30))
    fig,axes=plt.subplots(1, 3, figsize=(10, 5), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[1].imshow(img_blur1[:, :, ::-1])
    axes[2].imshow(img_blur2[:, :, ::-1])
    plt.show()

cv2_do()
print_hi('PyCharm')


