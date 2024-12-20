import cv2
import numpy as np
from data_ import getDataset
from torch.utils.data import DataLoader, Dataset
#from modelv2 import Encoder, Segnet
#from modelv3 import Segnet as SegnetSkip3
import torch
from train import train
#from model_dilated import SegmentationDilated as SegmentationDil
#from model_dilated2 import SegmentationDilated as SegmentationDil2
#from model_dilated3 import SegmentationDilated as SegmentationDil3
#from model_dilated4 import SegmentationDilated as SegmentationDil4
#from model_dilated5 import SegmentationDilated as SegmentationDil5
#from labels import mylabels, Label, id2myid, id2label
#from depthmodel import SwinTransformer, NeWCRFDepth
#from depthmodel2 import NeWCRFDepth as NeWCRFDepth2
#from depthmodel3 import NeWCRFDepth as NeWCRFDepth3
#from losses import depth_loss
from data_config import rawdatalist, depthdatalist, valrawdatalist, valdepthdatalist
from swin_transformer import SwinTransformer
from transformers import AutoImageProcessor, SwinModel
from torchsummary import summary
from UperNet import UperNet
from time import time
from load_pretrained import weights_init, load_pretrained_classification
import torch.nn as nn


kittidatapath = "C:\\Users\\Kush\\OneDrive\\Desktop\\CV-ML\\datasets\\data_semantics\\training"

#CamViddatapath = "C:\\Users\\Kush\\OneDrive\\Desktop\\CV-ML\\datasets\\SegNet-Tutorial-master\\SegNet-Tutorial-master\\CamVid"

#kittiCamViddataset = getDataset(datapath=[kittidatapath, CamViddatapath], pct=1.0, train_val_split=1.0, dataset='kittiCamVid', data_augment=False, gt_present=True, mode='train', resizex=224, resizey=224, shuffle=True, data_augment_flip=0.20, data_augment_brightness_color=0.20)

kittidataset = getDataset(datapath=[kittidatapath], pct=0.875, train_val_split=1.0, dataset='kitti_19', data_augment=False, gt_present=True, mode='train', resizex=224, resizey=224, shuffle=True, data_augment_flip=0.20, data_augment_brightness_color=0.20)


#resultsdir = 'results/trial1_UperNet_kittiCamVid'


#batch_size = 4

#batch_size_val = 4

#data_loader = DataLoader(kittidataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#data_loader = DataLoader(kittiCamViddataset, batch_size=batch_size, shuffle=False, pin_memory=True)


#datapathcam = 'C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/SegNet-Tutorial-master/SegNet-Tutorial-master/CamVid'

resizex = 224

resizey = 224

#dataset = getDataset(datapathcam, dataset='CamVid', data_augment=False, gt_present=True, mode='train', resizex=resizex, resizey=resizey)

#kittiCamViddataset = getDataset(datapath=[kittidatapath, CamViddatapath], pct=1.0, train_val_split=1.0, dataset='kittiCamVid', data_augment=False, gt_present=True, mode='train', resizex=224, resizey=224, shuffle=False)

#valdataset = getDataset(rawdatapath=valrawdatalist, depthdatapath=valdepthdatalist, max_depth=85, pct=0.30, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.0, gt_present=True, mode='train')

#resultsdir = 'results/trial2_UperNet_kittiCamVid'

resultsdir = 'results/trial3_UperNet_kitti19'


batch_size = 4

#batch_size_val = 4

#data_loader = DataLoader(kittidataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#data_loader2 = DataLoader(kittiCamViddataset, batch_size=batch_size, shuffle=False, pin_memory=True)

val_dataset = getDataset(datapath=[kittidatapath], dataset='kitti_19', data_augment=False, gt_present=True, mode='val', resizex=resizex, resizey=resizey)

print('num_classes: ', kittidataset.num_classes)

model = UperNet(kittidataset.num_classes, in_channels=3, backbone='swinv1_7_224', pretrained=True, use_aux=True, fpn_out=96, freeze_bn=False, head_out=96, backbonepath='swin_tiny_patch4_window7_224.pth')

#dataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.20, gt_present=True, mode='train', resizex=224, resizey=224)

#valdataset = getDataset(rawdatapath=valrawdatalist, depthdatapath=valdepthdatalist, max_depth=85, pct=0.30, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.0, gt_present=True, mode='train')

#resultsdir = 'results/trial_UperNet_CamVid'

criterion = nn.CrossEntropyLoss()

#batch_size = 4

batch_size_val = 4

for name, param in model.named_parameters():
    print('name: ', name)

print('model: ', model)


data_loader = DataLoader(kittidataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

val_data_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, drop_last=True)

print('len(data_loader.dataset): ', len(data_loader.dataset))

print('data_loader.batch_size: ', data_loader.batch_size)

train(data_loader, val_data_loader, model, criterion, epochs=35, batch_size=4, modelpath=None, bestmodelpath=None, resume_training=False, useWeights=False, resultsdir=resultsdir)

