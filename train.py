## Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method
## Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Bjorn Stenger, Tong Lu
## https://arxiv.org/pdf/2212.11548.pdf

import os
import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
import random
from transform.data_RGB import get_training_data,get_validation_data2
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import utils.losses
from model.LLFormer import LLFormer
import argparse
parser = argparse.ArgumentParser(description='Hyper-parameters for LLFormer')
parser.add_argument('-yml_path', default="./training.yaml", type=str)
args = parser.parse_args()


# Low Light Satellite Model imports
from utils import ECLLSIE_loss_functions as sat_loss # satellite model loss functions



## Set Seeds
torch.backends.cudnn.benchmark = True
#random.seed(1234)
#np.random.seed(1234)
#torch.manual_seed(1234)
#torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
yaml_file = args.yml_path

with open(yaml_file, 'r') as config:
    opt = yaml.safe_load(config)
print("load training yaml file: %s"%(yaml_file))

Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,4,8,16],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = False)
p_number = network_parameters(model)
model.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
# L1loss = nn.L1Loss()
#Charloss = nn.SmoothL1Loss()

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=8, drop_last=False)
val_dataset = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}
    GPU Name:           {'GPU' + torch.cuda.get_device_name(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

# instantiate losses
L_color   = sat_loss.L_color(8)
# L_color = Myloss.L_color(16)
L_spa     = sat_loss.L_spa()
L_exp     = sat_loss.L_exp(16)
# L_exp   = sat_loss.L_exp(16,0.6)

L_sat     = sat_loss.L_SAT(8)
L_TV      = sat_loss.L_TV()
L_col_con = sat_loss.L_col_con()
#L_sa     = sat_loss.Sa_Loss()

W_TV      = 20
W_spa     = 220
W_sat     = 10
W_col     = 10
W_exp     = 25
W_col_con = 60


for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    batches = 0
    train_id = 1

    model.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # Forward propagation
        for param in model.parameters():
            param.grad = None

        LL_img = data[0].cuda()
        enhanced_img = model(LL_img)

        E = 0.575 # TODO: Paper with 0.6, Walter with 0.575, Try 0.5

        #print(enhanced_img.shape,LL_img.shape)

        # Compute loss
        loss_spa = torch.mean(L_spa(enhanced_img, LL_img))
        loss_col = torch.mean(L_color(enhanced_img))
        loss_sat = torch.mean(L_sat(enhanced_img))

        loss_exp = torch.mean(L_exp(enhanced_img,E))
        loss_TV  = torch.mean(L_TV(enhanced_img))
        loss_col_con = torch.mean(L_col_con(LL_img, enhanced_img))

        
        '''
        print(f"Unweighted:\nTV: {loss_TV}, SPA: {loss_spa}, SAT: {loss_sat}, EXP: {loss_exp}, COL_CON: {loss_col_con}, COL: {loss_col}")
        print(f"Weighted:\nTV: {W_TV*loss_TV}, SPA: {W_spa*loss_spa}, SAT: {W_sat*loss_sat}, \
                EXP: {W_exp*loss_exp}, COL_CON: {W_col_con*loss_col_con}, COL: {W_col*loss_col}")
        '''
        
        #loss = W_TV*loss_TV + W_spa*loss_spa + W_sat*loss_sat + W_exp*loss_exp + W_col_con*loss_col_con + W_col*loss_col
        loss = torch.log(1 +
                W_TV * loss_TV +
                W_spa * loss_spa +
                W_sat * loss_sat +
                W_col * loss_col +
                W_exp * loss_exp +
                W_col_con * loss_col_con
                )

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

    '''
    ################################################################
    ##### TODO Probably necessary to rewrite validation entirely ###
    ################################################################
    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model.eval()
        psnr_val_rgb = []

        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[0].cuda()
            h, w = target.shape[2], target.shape[3]
            with torch.no_grad():
                enhanced_img = model(input_)
                enhanced_img = enhanced_img[:, :, :h, :w]

            for res, tar in zip(enhanced_img, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(enhanced_img, target))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR.pth"))
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

    # Save the best SSIM model of validation
    if ssim_val_rgb > best_ssim:
        best_ssim = ssim_val_rgb
        best_epoch_ssim = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_bestSSIM.pth"))
    print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
    epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))
    '''
    # Save evey epochs of model
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

    #writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
    #writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)

    scheduler.step()
    epoch_loss /= batches

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))
