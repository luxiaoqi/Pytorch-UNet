import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

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


def get_args():
    parser = argparse.ArgumentParser(description='convert json to image and label for labelme')
    parser.add_argument('inDir')
    parser.add_argument('outDir')
    return parser.parse_args()

if __name__== "__main__" :
    args = get_args()
    if len(args.inDir) == 0 or len(args.outDir) == 0:
        exit()
    path =  args.inDir #'D:\workspace\json'
    out =  args.outDir #"D:\workspace\jsonOut"
    files = getFiles(path, '.json')
    file = files[0]
    imagedir='image'
    labeldir='label'
    outImage = os.path.join(out, imagedir)
    if not os.path.exists(outImage):
        os.mkdir(outImage)
    outLabel = os.path.join(out, labeldir)
    if not os.path.exists(outLabel):
        os.mkdir(outLabel)

    for i in range(len(files)):
        # if i > 3:
        #     break
        file = files[i]
        cmd = 'labelme_json_to_dataset.exe "' + file + '" '
        file = os.path.basename(file)
        res = os.path.splitext(file)
        outfile = res[0]
        outfile = os.path.join(out,outfile)
        cmd = cmd + ' -o  "'+outfile+'"'
        os.system(cmd)

        # img_arr = cv2.imread(os.path.join(outfile,"label.png"), 0)
        # print(np.max(img_arr), np.min(img_arr))
        file = res[0]
        os.rename(os.path.join(outfile,"label.png"), os.path.join(outfile,file+'.png'))
        shutil.move(os.path.join(outfile, file + '.png'), os.path.join(outLabel, file + '.png'))

        os.rename(os.path.join(outfile, "img.png"), os.path.join(outfile, file + '.png'))
        shutil.move(os.path.join(outfile, file + '.png'), os.path.join(outImage, file + '.png'))

        #删除目录
        shutil.rmtree(outfile)

        #查看图片
        #cv2.imshow("image show",img_arr)
        #cv2.waitKey(0)
        #输出直方图
        # plt.hist(img_arr.ravel(), 256, [0, 256])
        # plt.show()


