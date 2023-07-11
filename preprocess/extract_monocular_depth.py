import cv2
import torch

import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
import numpy as np

parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')
parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

def save_outputs(img_path, output_file_name):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)
    save_path = os.path.join(args.output_path, f'{output_file_name}_depth.png')
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
        ).squeeze()

        output = prediction

        #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
        #output = output.clamp(0,1)
        #print(output.detach().cpu().squeeze().min(), output.detach().cpu().squeeze().max())
        #np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy()[0])
        output = 255 - output
        #output = standardize_depth_map(output)
        plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='gray')
            
        print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*.png'):
        if "_normal.png" in f or "depth.png" in f:
            continue
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()