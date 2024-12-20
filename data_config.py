import numpy as np


rawdatapath1 = "D:/kitti raw images/2011_09_28_drive_0002_sync/2011_09_28/2011_09_28_drive_0002_sync/image_02/data"

depthdatapath1 = "D:/data_depth_annotated/train/2011_09_28_drive_0002_sync/proj_depth/groundtruth/image_02"

rawdatapath2 = "D:/kitti raw images/2011_09_26_drive_0117_sync/2011_09_26/2011_09_26_drive_0117_sync/image_02/data"

depthdatapath2 = "D:/data_depth_annotated/train/2011_09_26_drive_0117_sync/proj_depth/groundtruth/image_02"

rawdatapath3 = "D:/kitti raw images/2011_09_26_drive_0052_sync/2011_09_26/2011_09_26_drive_0052_sync/image_02/data"

depthdatapath3 = "D:/data_depth_annotated/train/2011_09_26_drive_0052_sync/proj_depth/groundtruth/image_02"

rawdatapath4 = "D:/kitti raw images/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/image_02/data"

depthdatapath4 = "D:/data_depth_annotated/train/2011_09_29_drive_0004_sync/proj_depth/groundtruth/image_02"

rawdatapath5 = "D:/kitti raw images/2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/image_02/data"

rawdatapath6 = "D:/kitti raw images/2011_09_26_drive_0017_sync/2011_09_26/2011_09_26_drive_0017_sync/image_02/data"

depthdatapath5 = "D:/data_depth_annotated/train/2011_09_26_drive_0009_sync/proj_depth/groundtruth/image_02"

depthdatapath6 = "D:/data_depth_annotated/train/2011_09_26_drive_0017_sync/proj_depth/groundtruth/image_02"

valrawdatapath1 = "D:/kitti raw images/val/2011_09_26_drive_0013_sync/2011_09_26/2011_09_26_drive_0013_sync/image_02/data"

valdepthdatapath1 = "D:/data_depth_annotated/val/2011_09_26_drive_0013_sync/proj_depth/groundtruth/image_02"

valrawdatapath2 = "D:/kitti raw images/val/2011_09_28_drive_0037_sync/2011_09_28/2011_09_28_drive_0037_sync/image_02/data"

valdepthdatapath2 = "D:/data_depth_annotated/val/2011_09_28_drive_0037_sync/proj_depth/groundtruth/image_02"

valrawdatapath3 = "D:/kitti raw images/val/2011_09_29_drive_0026_sync/2011_09_29/2011_09_29_drive_0026_sync/image_02/data"

valdepthdatapath3 = "D:/data_depth_annotated/val/2011_09_29_drive_0026_sync/proj_depth/groundtruth/image_02"

#rawdatalist = [rawdatapath1, rawdatapath2, rawdatapath3, rawdatapath4, rawdatapath5]

#depthdatalist = [depthdatapath1, depthdatapath2, depthdatapath3, depthdatapath4, depthdatapath5]

rawdatalist = [rawdatapath1]

depthdatalist = [depthdatapath1]

valrawdatalist = [valrawdatapath1, valrawdatapath2, valrawdatapath3]

valdepthdatalist = [valdepthdatapath1, valdepthdatapath2, valdepthdatapath3]

def get_mean_std(dataset_name='kitti'):
	if dataset_name == 'kitti':
		return mean, std