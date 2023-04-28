import torch
import torchvision
from unet import UNet
import logging
import argparse
import os
from PIL import Image
from utils import BasicDataset


# model = torchvision.models.resnet18()
# # 生成一个样本供网络前向传播 forward()
# example = torch.rand(1, 3, 224, 224)
#
# # 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
# traced_script_module = torch.jit.trace(model, example)
#
# traced_script_module.save('model.pt')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='traceImage.png', metavar='FILE', help='Filenames of input images')
    parser.add_argument('--output', '-o', default='traceMODEL1.pth', metavar='FILE', help='Modelnames of output model')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    in_file = args.input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    if len(in_file):
        filename = in_file
        logging.info(f'Predicting image {filename} ...')
        if not os.path.exists(filename):
            exit()
        full_img = Image.open(filename)
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, (256, 256), args.scale,
                                                       is_mask=False))  # w, h = image_size #pil_img.size
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        traced_net = torch.jit.trace(net, img)
        output = traced_net(img).cpu()
        print(output)
        traced_net.save(args.output)


