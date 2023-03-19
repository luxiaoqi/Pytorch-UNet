import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from pathlib import *
import os



def del_dir(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        #print("root: ", root, "  dirs: ", dirs, "  files: ", files)
        '''
        root:  foo/bar/baz/empty/test   dirs:  []   files:  []
        root:  foo/bar/baz/empty   dirs:  ['test']   files:  []
        root:  foo/bar/baz   dirs:  ['empty']   files:  ['test_bak.txt', 'test.txt']
        '''
        #continue
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

# 创建文件夹函数
def creat_dir(dir_path):
    if not os.path.exists(dir_path):  # 如果不存在
        os.makedirs(dir_path)


	#if Path(dir_path).exists():  # 如果存在，则删除文件夹
	#	Path.rmdir(Path(dir_path))



class FeatureSaveToImage():
    def __init__(self, img_dir="out", enable=True):
        self.img_dir = img_dir
        self.enable = enable

    def save_feature_to_img(self, featureVec, img_path):
        # to numpy
        # feature=self.get_single_feature()

        if not self.enable:
            return
        img_path = self.img_dir+"\\"+img_path
        del_dir(img_path)
        creat_dir(img_path)
        for index, feature in enumerate(featureVec[0]):
            # feature = feature[:, 0, :, :]
            #feature = feature.view(feature.shape[1], feature.shape[2])
            feature = feature.cpu().numpy()
            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)
            # print(feature[0])
            # cv2.imwrite('./img.jpg',feature)
            cv2.imwrite(("%s//%d.jpg") %(img_path,index) , feature)


#creat_dir(Path("lxq/lxq.txt").parent)