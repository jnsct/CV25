## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import numpy as np

from torchvision.transforms import functional as TTF

from utils import ECLLSIE_loss_functions as sat_loss

import cv2
import argparse
from model.LLFormer import LLFormer
parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/ZDCE/test/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/ZDCE/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='./datasets/ZDCE/checkpoints/LLFormer_ZDCE/models/model_epoch_1.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()


# instantiate losses
L_color   = sat_loss.L_color(8)
L_spa     = sat_loss.L_spa()
L_exp     = sat_loss.L_exp(16)
L_tv      = sat_loss.L_TV()
E = 0.575

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                  + glob(os.path.join(inp_dir, '*.JPG'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.PNG')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights

model = LLFormer(inp_channels=3,out_channels=3,dim = 16,
                 num_blocks = [2,4,8,16],num_refinement_blocks = 2,
                 heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,
                 LayerNorm_type = 'WithBias',attention=True,skip = True)
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('enhancing images...')

mul = 16
index = 0
psnr_val_rgb = []

for file_ in files:
    img = Image.open(file_).convert('RGB')


    # ensure minimum size to prevent padding down to zero
    min_size = 64
    if img.size[0] < min_size or img.size[1] < min_size:
        print(f"Skipping {file_} — image too small: {img.size}")
        continue

    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        enhanced_img = model(input_)

    enhanced_img = torch.clamp(enhanced_img, 0, 1)
    enhanced_img = enhanced_img[:, :, :h, :w]
    enhanced_img = enhanced_img.permute(0, 2, 3, 1).cpu().detach().numpy()
    enhanced_img = img_as_ubyte(enhanced_img[0])

    print(f'img shape {enhanced_img.shape}')

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), enhanced_img)
    index += 1
    
    # float and channel first
    enhanced_img = torch.tensor(enhanced_img).float().permute(2, 0, 1).unsqueeze(dim=0).cuda() / 255.0
    refimg = TTF.pil_to_tensor(img).float().unsqueeze(dim=0).cuda() / 255.0

    loss_spa = 100*torch.mean(L_spa(enhanced_img, refimg))
    loss_col = torch.mean(L_color(enhanced_img))
    loss_exp = 50*torch.mean(L_exp(enhanced_img,E))
    loss_tv  = 10*torch.mean(L_tv(enhanced_img))

    print("loss_spa: ", round(loss_spa.item(),4), "\nloss_col: ", round(loss_col.item(),4), "\nloss_exp: ",round(loss_exp.item(),4),"\nloss_tv: ",round(loss_tv.item(),4))
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')



