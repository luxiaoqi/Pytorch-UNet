#!/usr/bin/env python

import argparse
import sys
import matplotlib.pyplot as plt
from labelme.label_file import LabelFile
from labelme import utils
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
#from labelme.logger import logger
import logging


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
    return filesTemp


##用来替换json中的图片
def replace_img():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("image_dir") #用于替换的图片所在位置
    args = parser.parse_args()
    image_dir = args.image_dir
    if not (args.json_dir and osp.isdir(args.json_dir)):
        return
    json_file_arr = getFiles(args.json_dir, ".json")
    for json_file in json_file_arr:
        #json_file = args.json_file
        data = json.load(open(json_file))
        #imageData = data.get("imageData")
        replacedImg = osp.splitext(osp.basename(json_file))[0]+".png"
        replacedImg = osp.join(image_dir, replacedImg)
        if not osp.exists(replacedImg):
            continue
        with open(replacedImg, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
        data["imageData"] = imageData
        data["imagePath"] = os.path.basename(replacedImg)
        out_json_file = os.path.splitext(json_file)[0]+".json"
        with open(out_json_file, "w") as f:
            json.dump(data, f)
            logging.info("Saved to: {}".format(out_json_file))


class reset_label_handler:
    def __init__(self, value=1):
        self.label_vaule = value

    def __call__(self, data, json_file):
        shapes = data["shapes"]
        for shape in shapes:
            shape["label"] = str(self.label_vaule)

def json_handler(handler):
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    #parser.add_argument("image_dir") #用于替换的图片所在位置
    args = parser.parse_args()
    #image_dir = args.image_dir
    if not (args.json_dir and osp.isdir(args.json_dir)):
        return
    json_file_arr = getFiles(args.json_dir, ".json")
    for json_file in json_file_arr:
        data = json.load(open(json_file))
        # data["imageData"] = imageData
        # data["imagePath"] = os.path.basename(replacedImg)
        if handler:
            handler(data, json_file)
        out_json_file = os.path.splitext(json_file)[0]+".json"
        with open(out_json_file, "w") as f:
            json.dump(data, f)
            logging.info("Saved to: {}".format(out_json_file))


#用来显示json文件结果
def draw_json():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    args = parser.parse_args()

    label_file = LabelFile(args.json_file)
    img = utils.img_data_to_arr(label_file.imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(label_file.shapes, key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, label_file.shapes, label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name
    lbl_viz = imgviz.label2rgb(
        lbl,
        imgviz.asgray(img),
        label_names=label_names,
        font_size=30,
        loc="rb",
    )

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()


if __name__ == "__main__":
    replace_img()  #替换json中的图片

    # handler = reset_label_handler(1)
    # json_handler(handler)

    #draw_json()
