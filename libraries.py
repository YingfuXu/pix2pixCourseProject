from __future__ import print_function
import os
from os import listdir
from os.path import join
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import functools
from matplotlib import pyplot as plt
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


def tensor_to_np(tensor):
    img = tensor.mul(1).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img