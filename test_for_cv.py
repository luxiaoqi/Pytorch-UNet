# This is a sample Python script.

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
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

    # img = Image.open('MainFrame_HDLogo.png')
    # ar = np.array(img)
    # img.show()
    pass

    # img_arr = cv2.imread('MainFrame_HDLogo.png')  # 'image\\11_101_29.png'
    # img_arr1 = cv2.imread('MainFrame_HDLogo1.png')
    # #img_arr = np.zeros((255, 255, 3),dtype=np.uint8)
    # cv2.imshow('one', img_arr)
    # cv2.imshow('one2', img_arr1)
    # #print(img_arr, '\n', img_arr.item(3, 6, 2))
    # cv2.waitKey(20000)

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
    # img = cv2.imread('image\\lenacolor.png')
    # img_blur1=cv2.blur(img, (3,3))
    # img_blur2=cv2.blur(img, (30,30))
    # fig,axes=plt.subplots(1, 3, figsize=(10, 5), dpi=100)
    # axes[0].imshow(img[:, :, ::-1])
    # axes[1].imshow(img_blur1[:, :, ::-1])
    # axes[2].imshow(img_blur2[:, :, ::-1])
    # plt.show()

    # #图像腐蚀
    # img = cv2.imread('image\\lenacolor.png', 0)
    # _, img = cv2.threshold(img, 127, 1, type=cv2.THRESH_BINARY)
    # fig,axes=plt.subplots(1, 4, figsize=(10, 5), dpi=100)
    # kernel1 = np.ones((5,5), np.uint8)
    # kernel2 = np.ones((10, 10), np.uint8)
    # img_erode_kernel1 = cv2.erode(img, kernel1)
    # img_erode_kernel2 = cv2.erode(img, kernel2)
    # axes[0].imshow(img, cmap='gray')
    # axes[1].imshow(img_erode_kernel1, cmap='gray')
    # axes[2].imshow(img_erode_kernel2, cmap='gray')
    # plt.show()

    #Canny边缘检测
    # lena = cv2.imread('image\\lenacolor.png', 0)
    # lena_canny_1 = cv2.Canny(lena, 128, 200)
    # lena_canny_2 = cv2.Canny(lena, 32, 128)
    # fig, axes = plt.subplots(1, 3, figsize=(10, 6), dpi=100)
    # axes[0].imshow(lena, cmap='gray')
    # axes[1].imshow(lena_canny_1, cmap='gray')
    # axes[2].imshow(lena_canny_2, cmap='gray')
    # plt.show()

    #下采样
    # lena0 = cv2.imread('image\\lenacolor.png', 0)
    # lena1 = cv2.pyrDown(lena0)
    # lena2 = cv2.pyrDown(lena1)
    # lena3 = cv2.pyrDown(lena2)
    # lena4 = cv2.pyrDown(lena3)
    # Fig = plt.figure(figsize=(16, 10))
    # Grid = plt.GridSpec(33, 33)
    # axes1 = Fig.add_subplot(Grid[:17, :17]), plt.imshow(lena0, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes2 = Fig.add_subplot(Grid[:9, 17:25]), plt.imshow(lena1, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes3 = Fig.add_subplot(Grid[:5, 25:29]), plt.imshow(lena2, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes4 = Fig.add_subplot(Grid[:3, 29:31]), plt.imshow(lena3, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes5 = Fig.add_subplot(Grid[:1, 31:32]), plt.imshow(lena4, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # plt.show()

    #上采样
    # lena0 = cv2.imread(r'image\\lenacolor.png', 0)  # 512
    # lena0 = cv2.resize(lena0, dsize=None, fx=1.0/2, fy=1.0/2)
    # lena1 = cv2.pyrUp(lena0)  # 1024
    # lena2 = cv2.pyrUp(lena1)  # 2048
    # lena3 = cv2.pyrUp(lena2)  # 4096
    # Fig = plt.figure(figsize=(16, 10))
    # Grid = plt.GridSpec(16, 16)
    # axes1 = Fig.add_subplot(Grid[0, 0]), plt.imshow(lena0, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes2 = Fig.add_subplot(Grid[0:3, 1:3]), plt.imshow(lena1, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes3 = Fig.add_subplot(Grid[0:5, 3:7]), plt.imshow(lena2, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes4 = Fig.add_subplot(Grid[0:9, 7:15]), plt.imshow(lena3, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # plt.show()

    #先下采样再上采样
    # lena0 = cv2.imread(r'image\\lenacolor.png', 0)  # 512
    # lena1 = cv2.pyrDown(lena0)  # 216
    # lena2 = cv2.pyrUp(lena1)  # 512
    # diff1 = lena2 - lena0
    # lena11 = cv2.pyrUp(lena0)  # 1024
    # lena22 = cv2.pyrDown(lena11)  # 512
    # diff2 = lena22 - lena0
    # Fig = plt.figure(figsize=(16, 10))
    # Grid = plt.GridSpec(6, 10)
    # axes1 = Fig.add_subplot(Grid[:2, :2]), plt.imshow(lena0, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes2 = Fig.add_subplot(Grid[0, 2]), plt.imshow(lena1, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes3 = Fig.add_subplot(Grid[:2, 3:5]), plt.imshow(lena2, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes4 = Fig.add_subplot(Grid[:2, 5:7]), plt.imshow(diff1, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes21 = Fig.add_subplot(Grid[2:4, :2]), plt.imshow(lena0, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes22 = Fig.add_subplot(Grid[2:6, 2:6]), plt.imshow(lena11, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes23 = Fig.add_subplot(Grid[2:4, 6:8]), plt.imshow(lena22, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # axes24 = Fig.add_subplot(Grid[2:4, 8:10]), plt.imshow(diff2, cmap='gray'), plt.box(), plt.xticks([]), plt.yticks([])
    # plt.show()

    #绘制图像内的轮廓
    # img = cv2.imread(r'image\\lenacolor.png', 0)  # 读图像
    # _,img = cv2.threshold(img, 127, 255, type=cv2.THRESH_BINARY)
    # kernel2 = np.ones((10, 10), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    # img_copy = img.copy()
    # img_gray = img_copy #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像变成灰度图像
    # t, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 将灰度图像变成二值图像
    # contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 生成轮廓
    # img_contours =  np.zeros(img_copy.shape, np.uint8)
    # img_contours = cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 10)  # 在原图上绘制轮廓
    # # 在一张黑色的背景里，分别绘制三个轮廓：
    # list = ['img_contours0', 'img_contours1', 'img_contours2']
    # for i in range(3):
    #     img_temp = np.zeros(img.shape, np.uint8)
    #     list[i] = cv2.drawContours(img_temp, contours, i, (0, 255, 0), 5)
    # # 可视化轮廓
    # fig, axes = plt.subplots(1, 4, figsize=(10, 6), dpi=100)
    # axes[0].imshow(img_contours, cmap='gray')
    # axes[1].imshow(list[0], cmap='gray')
    # axes[2].imshow(img, cmap='gray')
    # axes[3].imshow(list[2], cmap='gray')
    # # axes[2].imshow(list[1], cmap='gray')
    # # axes[3].imshow(list[2], cmap='gray')
    # plt.show()

    # img = cv2.imread(r'image\\lenacolor.png')
    # mask = np.zeros(img.shape, np.uint8)
    # mask[200:400, 200:400] = 255
    # img_mask = cv2.bitwise_and(img, mask)
    # hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist_img_mask = cv2.calcHist([img], [0], mask[:, :, 0], [256], [0, 256])
    # hist_mask = cv2.calcHist([img_mask[200:400, 200:400]], [0], None, [256], [0, 256])
    # # 可视化
    # plt.figure(figsize=(12, 3))
    # plt.subplot(151), plt.imshow(img[:, :, ::-1])
    # plt.subplot(152), plt.imshow(img_mask[:, :, ::-1])
    # plt.subplot(153), plt.plot(hist_img), plt.plot(hist_img_mask)  # 无掩膜和有掩膜的直方图画到一起
    # plt.subplot(154), plt.plot(hist_img_mask)  # 单独划出有掩膜的直方图
    # plt.subplot(155), plt.plot(hist_mask)  # 单独把mask部分图像的直方图画出来，和上面的一模一样
    # plt.show()

    #######分水岭算法
    # img = cv2.imread(r'water_coins.jpg')  # img.shape返回：(312, 252, 3)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # img_gray.shape返回：(312, 252)
    # # ------------Otsu阈值处理,变成二值图像--------------
    # t, otsu = cv2.threshold(img_gray, 0, 255,
    #                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # t返回162.0， otsu.shape返回(312, 252)
    # # ------------形态学的开运算，就是先腐蚀erode后膨胀dilate,目的一是去噪，二是先把前景对象重叠的部分分开，方便后面计数或者画每个对象的轮廓-------------
    # img_opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)  # A 这还是一个二值图像
    # # -------计算距离,确定前景对象--------------------
    # dist = cv2.distanceTransform(img_opening, cv2.DIST_L2, 5)  # float32的浮点型数组，dist.shape返回(312, 252),dist是一个灰度图像
    # th, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255,
    #                             cv2.THRESH_BINARY)  # 把dist阈值化处理，变成一个0和255的二值图像，此时就是我们要的确定前景
    # sure_fg = np.uint8(sure_fg)
    # # -----计算确定背景、计算未知区域------------------
    # sure_bg = cv2.dilate(img_opening, kernel=np.ones((3, 3), np.uint8), iterations=3)  # 对前景膨胀来确定背景
    # unknown = cv2.subtract(sure_bg, sure_fg)  # 确定背景图-确定前景图，生成未知区域图
    # # ------标注确定前景图,调整标注规则---------------------------
    # ret, labels = cv2.connectedComponents(sure_fg)  # 有24个硬币，ret返回是25, labels是一个形状是(312, 252)的int32的数组
    # labels = labels + 1  # 把背景标为1，前景对象依次为2，3，，，26
    # labels[unknown == 255] = 0  # 0代表未知区域
    # loc = np.where(labels == 0)
    # # ------------使用分水岭算法对图像进行分割---------------
    # img1 = img.copy()
    # labels1 = labels.copy()
    # markers = cv2.watershed(img1, labels)
    # img1[markers == -1] = [0, 255, 0]
    # # 可视化：
    # plt.figure(figsize=(12, 6))
    # plt.subplot(251), plt.imshow(img[:, :, ::-1])  # 原图
    # plt.subplot(252), plt.imshow(img_gray, cmap='gray')  # 灰度图
    # plt.subplot(253), plt.imshow(otsu, cmap='gray')  # otsu阈值处理后的二值图
    # plt.subplot(254), plt.imshow(img_opening, cmap='gray')  # 开运算去噪后的图像
    # plt.subplot(255), plt.imshow(dist, cmap='gray')  # 距离图像
    # plt.subplot(256), plt.imshow(sure_fg, cmap='gray')  # 确定前景
    # plt.subplot(257), plt.imshow(sure_bg, cmap='gray')  # 确定背景
    # plt.subplot(258), plt.imshow(unknown, cmap='gray')  # 确定未知区域图
    # plt.subplot(259), plt.imshow(labels, cmap='gray')  # 标注图
    # plt.subplot(2, 5, 10), plt.imshow(img1[:, :, ::-1])  # 分割结果
    # plt.show()
    #import normalize_thorax_PA as thorax
    #print(thorax.settings)


cv2_do()
print_hi('PyCharm')


