#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from net import *
import torch.nn.functional as F
from utils import ReplayBuffer
from utils import LambdaLR
#from utils import Logger
from tqdm import tqdm
#from utils import weights_init_normal
from datasets import PairClearDepth
import os
import numpy as np
import cv2
import torch.nn.functional as f

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0")
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True


batchSize = 1
n_cpu = 0


netG_B2A = Generator(3,3).eval()
netG_B2A.load_state_dict(torch.load(r"C:\Users\Admin\Desktop\paper\code\weight\265_1.3175, _1.1272, .pth"))

netG_B2A.to(device)





# Dataset loader

               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(PairClearDepth("test"),batch_size=batchSize, shuffle=True, num_workers=n_cpu)


if __name__ == "__main__":

        factor = 4
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        #torch.save(netG_B2A.state_dict(),r"C:\Users\Admin\Desktop\paper\code\Ablation study\result3\373_1.9984, _1.4911, .pth")
        with torch.no_grad():
            for i, batch in loop:
                # Set model input
                clearImage = batch['clearImage'].to(device)
                #depthImage = batch['depthImage'].to(device)
                HazeImage = batch['HazeImage'].to(device)
                ImageName = batch['ImageName']

                h, w = HazeImage.shape[2],HazeImage.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                fake_Haze = f.pad(HazeImage, (0, padw, 0, padh), 'reflect')









                max_values_1, _ = torch.max(fake_Haze, dim=1)
                max_values_2, _ = torch.max(max_values_1, dim=1)
                max_values, _ = torch.max(max_values_2, dim=1)
                A = max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3).detach()

                transmap = netG_B2A(fake_Haze)
                #transmap = (A - fake_Haze[:, :, :, :] + transmap) / A + 0.001
                #transmap = (A - fake_Haze + transmap) / A + 0.001
                recovered_Clear = (fake_Haze - A * (1 - transmap)) / transmap
                recovered_Clear = transmap[:, :, :HazeImage.shape[2], :HazeImage.shape[3]]

                tensor_example2 = recovered_Clear[:1, :3, :, :].squeeze(0).clamp(0, 1) * 255  # (tensor_example2 - tensor_example2.min()) / (tensor_example2.max() - tensor_example2.min()) * 255
                tensor_example2 = tensor_example2.byte()
                image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
                image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)

                cv2.imwrite(r"C:\Users\Admin\Desktop\paper\code\Ablation study\try\\"+ImageName[0],image_array2)

                # cv2.imshow("dehaze", image_array2)
                # #
                # tensor_example2 = fake_Haze[:1, :3, :, :].squeeze(0) * 255  # (tensor_example2 - tensor_example2.min()) / (tensor_example2.max() - tensor_example2.min()) * 255
                # tensor_example2 = tensor_example2.byte()
                # image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
                # image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)
                # cv2.imshow("Haze", image_array2)
                # #
                # #
                # #
                # #
                # #
                # tensor_example2 = clearImage[:1, :3, :, :].squeeze(
                #     0) * 255  # (tensor_example2 - tensor_example2.min()) / (tensor_example2.max() - tensor_example2.min()) * 255
                # tensor_example2 = tensor_example2.byte()
                # image_array2 = np.array(tensor_example2.cpu().permute(1, 2, 0))
                # image_array2 = cv2.cvtColor(image_array2, cv2.COLOR_RGB2BGR)
                # cv2.imshow("clearImage", image_array2)
                # cv2.waitKey(0)







