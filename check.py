import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import cv2
from PIL import Image


checkpoint = torch.load('resnet18_baseline_1000.pth')
print([k for k in checkpoint.keys() if 'fc' in k])