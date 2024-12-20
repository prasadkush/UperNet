# UperNet

<br/>
This is an implementation of a training pipeline for semantic segmentation of images and UperNet [1] (using Swin Transformer [2] as the backbone) code in PyTorch. The model was trained on the outdoor road scenes CamVid dataset.
<br/><br/>

## Instructions for training:

<br/>
Change the values of path of dataset and change the data_.py file if needed (will be needed if using a dataset other than kitti or CamVid).
<br/><br/><br/>


```
 Run python main.py.
```

<br/>

## Results on CamVid dataset 

<br/>
The model was trained on the CamVid dataset consisting of 367 images for training and 101 images for validation. The images were resized to 224 x 224 and pretrained weights of imagenet 1000 were used for the swin transformer backbone. The model was trained for 70 epochs.
<br/><br/></br>


| Metric  | Value |
| --- | --- |
| pixel accuracy on validation dataset| 90.39 % |
| mean IoU on validation dataset | 57.38 % |
<br/>


<br/>
Below are results on some images of the CamVid validation dataset. leftmost is the segmented output, middle is the original image resized to 224 x 224 resolution and rightmost is the segmentation overlayed on original RGB image. The model and was trained on original images and ground truth resized to 224 x 224 resolution. 
<br/><br/>

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/10.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/101.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/108.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/111.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/124.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/133.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/136.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/42.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/46.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/47.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/49.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/52.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/55.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/56.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/58.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/6.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/65.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/67.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/68.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/70.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/78.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/85.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/UperNet/blob/main/CamVid%20results/96.jpg)

<br/><br/>
## Acknowledgements

<br/><br/>
Some code for UperNet was taken from MMsegmentation (Copyright 2020 The MMSegmentation Authors), OpenMMLab Semantic Segmentation Toolbox and Benchmark (https://github.com/open-mmlab/mmsegmentation) and PyTorch UperNet implementation and modified a bit. Some code for Swin Trransformer was taken from Microsoft's official PyTorch implementation of Swin Transformer (https://github.com/microsoft/Swin-Transformer) Copyright (c) Microsoft Corporation and https://github.com/berniwal/swin-transformer-pytorch and modified.
<br/><br/>

## References

<br/>
[1] Xiao, Tete, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. "Unified perceptual parsing for scene understanding." In Proceedings of the European conference on computer vision (ECCV), pp. 418-434. 2018.

[2] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

[3] https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py

[4] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py

[5] https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/psp_head.py

[6] https://github.com/berniwal/swin-transformer-pytorch

[7] https://github.com/microsoft/Swin-Transformer

[8] https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678

[9] https://medium.com/thedeephub/building-swin-transformer-from-scratch-using-pytorch-hierarchical-vision-transformer-using-shifted-91cbf6abc678





