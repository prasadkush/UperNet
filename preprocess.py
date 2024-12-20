import cv2
import numpy as np
from data_ import getDataset
from torch.utils.data import DataLoader, Dataset
import torch
from itertools import cycle

kittidatapath = "C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/data_semantics/training"

CamViddatapath = "C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/SegNet-Tutorial-master/SegNet-Tutorial-master/CamVid"


def get_mean(datapath=None, dataset='kitti'):
	#datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training'
	#datapathcam = 'C:/Users/Kush/OneDrive/Desktop/CV-ML/datasets/SegNet-Tutorial-master/SegNet-Tutorial-master/CamVid'
	dataset = getDataset(datapath=datapath, dataset=dataset, resizex=224, resizey=224, shuffle=False, pct=1.0, train_val_split=1.0)
	size_data = len(dataset)
	loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
	loaderiter = iter(loader)
	imgh = 0
	imgw = 0
	meanimg = 0
	print('size_data: ', size_data)
	for i in range(size_data):
		data = next(loaderiter)
		d = data['image'].numpy()[0,:,:,:]
		d = d.transpose((1,2,0))
		if i == 0:
			imgh = d.shape[0]
			imgw = d.shape[1]
			print('imgh: ', imgh)
			print('imgw: ', imgw)
			meanimg = np.zeros((imgh,imgw,3))
		#if i == 0 or i == 1:
		#	print('mean i: ', i)
		#	print('d[0:100,0:100,0]: ', d[0:100,0:100,0])
		meanimg = (i*meanimg + d)/(i+1)
	simg = np.sum(size_data*meanimg, axis=0)
	simg = np.sum(simg, axis = 0)
	#print('simg: ', simg)
	mean = simg/(size_data*imgh*imgw)
	print('mean: ', mean)
	return mean, dataset, loader
	#print('mean/255: ', mean/255)


def get_std(datapath=None, mean=None, dataset='kitti'):
	#datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training'
	mean, dataset, loader = get_mean(datapath, dataset=dataset)
	loaderiter = iter(cycle(loader))
	size_data = len(dataset)
	imgh = 0
	imgw = 0
	meanstdimg = 0
	for i in range(size_data):
		data = next(loaderiter)
		d = data['image'].numpy()[0,:,:,:]
		d = d.transpose((1,2,0))
		if i == 0:
			imgh = d.shape[0]
			imgw = d.shape[1]
			meanstdimg = np.zeros((imgh, imgw, 3))
		#if i == 0 or i == 1:
		#	print('std i: ', i)
		#	print('d[0:100,0:100,0]: ', d[0:100,0:100,0])
		#	print('d - mean: ', (d - mean)[0:5,0:5,:])
		#	print('mean: ', mean)
		stdimg = d - mean
		#if i == 0:
		#print('stdimg[0:100, 0:100, 0]: ', stdimg[0:100, 0:100, 0])
		stdimg = stdimg**2
		#if i == 0:
		#print('after square')
		#print('stdimg[0:100, 0:100, 0]: ', stdimg[0:100, 0:100, 0])
		meanstdimg = (i*meanstdimg + stdimg)/(i+1)

	sumstdimg = np.sum(size_data*meanstdimg, axis = 0)
	sumstdimg = np.sum(sumstdimg, axis = 0) 
	std = sumstdimg/(size_data*imgh*imgw)
	std = np.sqrt(std)
	print('std: ', std)
	return std, mean

def get_mean_std(dataset_name='kitti'):
    if dataset_name == 'kitti' or dataset_name == 'kitti_19':
    	mean = np.array([0.38399986, 0.39878138, 0.3793309 ])
    	std  = np.array([0.32906724, 0.31968708, 0.31093021])
    elif dataset_name == 'CamVid':
    	mean = np.array([0.41491247, 0.41524811, 0.39985576])
    	std = np.array([0.31305884, 0.31191743, 0.30566974])
    return mean, std

#std, mean = get_std(datapath=[kittidatapath, CamViddatapath], dataset = 'kittiCamVid')


#for kitti
#mean:  [0.38399986 0.39878138 0.3793309 ]
#std:  [0.32906724 0.31968708 0.31093021]

#CamVid
#std:  [0.30721843 0.31161108 0.3070735 ]
#mean:  [0.4326707  0.4251328  0.41189488]
