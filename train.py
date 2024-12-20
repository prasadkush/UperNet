import cv2
import torch
import torch.nn as nn
from torch import optim
from data_ import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from UperNet import UperNet
import matplotlib.pyplot as plt
import numpy as np
from config import datapath, datapathcam
import pickle
import pdb
from predict import predict_single_image, compute_accuracy, compute_intersection_union
from Exceptions import ModelPathrequiredError
from torchvision.models import vgg16_bn
from utils import find_nth_occurence
import random
from time import time

weight_dict = {}

def save_model(loss, path, epoch, model, optimizer, lr_schedule, lr_milestones, mean_iou):
    EPOCH = epoch
    PATH_ = path
    LOSS_ = loss
    MEAN_IOU_ = mean_iou

    torch.save({
        'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS_, 'mean_iou': MEAN_IOU_, 'lr_schedule': lr_schedule, 'lr_milestones': lr_milestones}, PATH_)

def compute_loss(dataset):
    batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size,shuffle=True)
    loaderiter = iter(data_loader)
    data = next(loaderiter)


def weights_init(m):
    #print('m: ', m)
    print('m.__class__: ', m.__class__)
    classname = m.__class__.__name__
    print('initializing weights, classname: ', classname)
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        #torch.nn.init.uniform_(m.weight)  
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)
    


def train(data_loader, val_data_loader, model, criterion, epochs=35, batch_size=4, modelpath=None, bestmodelpath=None, resume_training=False, useWeights=False, resultsdir=None):

    # TODO initialize this to be a Cross Entropy Classification loss.
    if useWeights == True:
        weights = dataset.getWeights()
        weights = weights.to(torch.float)
        print('weights: ', weights)
        criterion = nn.CrossEntropyLoss(weight=weights)
        #criterion = nn.NLLLoss(weight=weights,reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss(reduction='none')


    lr_initial = 0.03
    lr_new = 0.03

    #optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=lr_initial,  momentum=0.9, nesterov=True)

    loaderiter_val = iter(val_data_loader)
    num_val = 0
    compute_val_over_whole_data = True

    start_epoch = 0
    loss = 0
    best_loss = 100000
    best_val_loss = 10000
    best_mean_iou = 0
    total_loss = 0
    training_loss_list = []
    #mean_list = []
    pixelacclist = []
    mean_ioulist = []
    pixelacc_val_list = []
    meaniou_val_list = []
    loss_val_list = []
    epochs = 60
    lr_schedule = [0.005, 0.001]
    lr_milestones = [30, epochs]


    if resume_training == True and modelpath != None:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if bestmodelpath != None:
            checkpoint = torch.load(bestmodelpath)
            best_loss = checkpoint['loss']
        else:
            best_loss = checkpoint['loss']
        start_epoch = epoch + 1
        ind_sch = np.searchsorted(np.array(lr_milestones), start_epoch, side='right')
        lr_new = lr_schedule[ind_sch]
        lr_initial = lr_schedule[ind_sch]
        print('start_epoch: ', start_epoch)
        print('ind_sch: ', ind_sch)
        print('lr_new: ', lr_new)

        for g in optimizer.param_groups:
            g['lr'] = lr_initial
            print('lr: ', g['lr']) 

        bestvalmodelpath = resultsdir + '/bestvallosssegnetmodelnew.pt'
        valcheckpoint = torch.load(bestvalmodelpath)
        valepoch = valcheckpoint['epoch']
        best_val_loss = valcheckpoint['loss']

        bestvalioumodelpath = resultsdir + '/bestmeaniousegmentmodelnew.pt'
        valioucheckpoint = torch.load(bestvalioumodelpath)
        iouvalepoch = valioucheckpoint['epoch']
        best_mean_iou = valioucheckpoint['mean_iou']


        with open(resultsdir + '/training_loss_list.pkl', 'rb') as f:
            training_loss_list = pickle.load(f)
            training_loss_list = training_loss_list[0:min(start_epoch,len(training_loss_list))]
            print('len(training_loss_list): ', len(training_loss_list))
            print('training_loss_list: ', training_loss_list)
            best_loss = min(training_loss_list)
        #with open(resultsdir + '/mean_list.pkl', 'rb') as f:
        #    mean_list = pickle.load(f)
        with open(resultsdir + '/pixelacclist.pkl', 'rb') as f:
            pixelacclist = pickle.load(f)
            pixelacclist = pixelacclist[0:min(start_epoch,len(pixelacclist))]
            print('len(pixelacclist): ', len(pixelacclist))
        with open(resultsdir + '/mean_ioulist.pkl', 'rb') as f:
            mean_ioulist = pickle.load(f)
            mean_ioulist = mean_ioulist[0:min(start_epoch,len(mean_ioulist))]
            print('len(mean_ioulist): ', len(mean_ioulist))
        with open(resultsdir + '/meaniou_val_list.pkl', 'rb') as f:
            meaniou_val_list = pickle.load(f)
            meaniou_val_list = meaniou_val_list[0:min(start_epoch,len(meaniou_val_list))]
            best_mean_iou2 = max(meaniou_val_list)
            print('len(meaniou_val_list): ', len(meaniou_val_list))
        with open(resultsdir + '/pixelacc_val_list.pkl', 'rb') as f:
            pixelacc_val_list = pickle.load(f)
            pixelacc_val_list = pixelacc_val_list[0:min(start_epoch,len(pixelacc_val_list))]
            print('len(pixelacc_val_list): ', len(pixelacc_val_list))
        with open(resultsdir + '/loss_val_list.pkl', 'rb') as f:
            loss_val_list = pickle.load(f)
            loss_val_list = loss_val_list[0:min(start_epoch,len(loss_val_list))]
            print('len(loss_val_list): ', len(loss_val_list))
            best_val_loss2 = min(loss_val_list)
            #best_val_loss = min(loss_val_list)
    elif resume_training == True and modelpath == None:
        raise ModelPathrequiredError("Provide Model path if resume_training is set to True")

    params_to_print1 = ['backbone.layers.0.blocks.0.mlp.fc1.weight', 'backbone.layers.1.blocks.1.mlp.fc1.weight', 'backbone.layers.2.blocks.1.norm1.weight', 'backbone.layers.3.blocks.1.attn.qkv.weight', 'PPMhead.bottleneck.0.weight',
     'FPN.conv1x1.0.weight', 'FPN.conv1x1.1.weight', 'head.layer1.0.weight', 'head.layer2.1.weight', 'ClassifyBlock.layer.weight']
    paramnames = [p[0:min(find_nth_occurence(p,'.',n=6),len(p))] for p in params_to_print1]
    paramshortlist = []
    wt_values = []
    prev_wt_values = []
    diff_wts = []
    for p in params_to_print1:
        paramshortlist.append(model.get_parameter(p))
        if len(paramshortlist[-1].data.shape) <= 2:
            wt_values.append(paramshortlist[-1].data)
        elif len(paramshortlist[-1].data.shape) == 4:
            wt_values.append(paramshortlist[-1].data[:,:,0,0])
        #elif len(paramshortlist[-1].data.shape) == 3:
        #    wt_values.append(paramshortlist[-1].data[:,:,0,])
        prev_wt_values.append(torch.zeros(wt_values[-1].shape))
        diff_wts.append(0)

    print('resume_training: ', resume_training)
    print('start_epoch: ', start_epoch)
    print('best_loss: ', best_loss)
    print('best_val_loss: ', best_val_loss)
    print('best_mean_iou: ', best_mean_iou)
    if resume_training == True:
        print('best_mean_iou2: ', best_mean_iou2)
        print('best_val_loss2: ', best_val_loss2)
        print('valepoch: ', valepoch)
        print('iouvalepoch: ', iouvalepoch)
        print('ind_sch: ', ind_sch)

    #epochs = 46
    #lr_schedule = [lr_initial, lr_initial/2, lr_initial/5, lr_initial/10]
    #lr_milestones = [10, 20, 30, epochs]
    #lr_schedule = [lr_initial, 0.01, 0.005, 0.001]
    #lr_milestones = [12, 24, 34, epochs]
    for g in optimizer.param_groups:
                g['lr'] = lr_new
    print('lr_schedule: ', lr_schedule)
    print('lr_milestones: ', lr_milestones)
    print('lr_new: ', lr_new)
    print('resultsdir: ', resultsdir)
    #loaderiter = iter(loader)
    if resume_training == False:
        ind_sch = 0

    for e in range(start_epoch, epochs):
        print('epoch: ', e)

        for g in optimizer.param_groups:
            print('g[lr]: ', g['lr'])

        if e in lr_milestones:
            ind_sch += 1
            lr_new = lr_schedule[ind_sch]
            for g in optimizer.param_groups:
                g['lr'] = lr_new

        total_loss = 0
        model.train()
        numimgs = 0
        pixelacc = 0
        intersect = np.zeros((model.num_classes,))
        union = np.zeros((model.num_classes,))
        for i, data in enumerate(data_loader):

            print('epoch: ', e, ' i: ', i)
            start = time()
            d = data['image']
            ds = data['semantic']
            dimg = data['original']
            optimizer.zero_grad()
            x = model.forward(d)
            #dimg = torch.permute(dimg, (0,2,3,1))
            #dimg = dimg.numpy()
            #print('dimg shape: ', dimg.shape)
            #cv2.imshow('dimg[0,:,:,:]: ', dimg[0,:,:,:])
            #cv2.waitKey(0)
            #inv_trans = get_inverse_transforms()
            #dorig = inv_trans(d)
            #dorig = torch.permute(dorig, (0,2,3,1))
            #dorig = dorig.numpy()
            #print('dorig shape: ', dorig.shape)
            #cv2.imshow('dorig[0,:,:,:]: ', dorig[0,:,:,:])
            #cv2.waitKey(0)
            print('x shape: ', x.shape)
            #print('x: ', x)
            with torch.no_grad():
                inds = torch.argmax(x, dim=1)
                pixelacc += np.sum(np.equal(inds.numpy(), ds.numpy()))/(inds.shape[1]*inds.shape[2])
                compute_intersection_union(inds, ds, model.num_classes, intersect, union)
                numimgs += x.shape[0]            
            x_logits = torch.logit(x,eps=1e-7) 
            #print('x_logits.shape')
            output = criterion(x_logits,ds)
            #output = criterion(x,ds)
            print('output: ', output)
            #breakpoint()
            output.backward(retain_graph=False)
            optimizer.step()
            loss = output.detach().item()
            total_loss = total_loss + batch_size*loss
            for l, param in enumerate(paramshortlist):
                #print('wt_values[l] norm: ', torch.norm(wt_values[l], p=2)/torch.numel(wt_values[l]))
                #wt_values[l] = param.data
                #print('wt_values[l] norm: ', torch.norm(wt_values[l], p=2)/torch.numel(wt_values[l]))
                diff_wts[l] = wt_values[l] - prev_wt_values[l]
                prev_wt_values[l].copy_(wt_values[l])
                if param.grad != None:
                    ix =  random.randint(0,param.grad.shape[0] - 6)
                    if len(param.grad.shape) == 2:
                        iy = random.randint(0,param.grad.shape[1] - 1)
                        print('grad ', l, ': ', param.grad[ix:ix+5,iy])
                        print('grad norm ', l, ': ', torch.norm(param.grad, p=1).item()/torch.numel(param.grad), '              name: ', paramnames[l])
                    elif len(param.grad.shape) == 4:
                        iy = random.randint(0,param.grad.shape[1] - 1)
                        print('grad ', l, ': ', param.grad[ix:ix+5,iy,0,0])
                        print('grad norm ', l, ': ', torch.norm(param.grad[:,:,0,0], p=1).item()/torch.numel(param.grad[:,:,0,0]), '              name: ', paramnames[l])
                    elif len(param.grad.shape) == 1:
                        print('grad ', l, ': ', param.grad[ix:ix+5])
                        print('grad norm ', l, ': ', torch.norm(param.grad, p=1).item()/torch.numel(param.grad), '              name: ', paramnames[l])                                        
                else:
                    print('grad ', l, ': ', param.grad)
                print('diff_wts[l] norm: ', torch.norm(diff_wts[l], p=1).item()/torch.numel(diff_wts[l]), '     wt norm: ', torch.norm(wt_values[l], p=1).item()/torch.numel(wt_values[l]))
            #if i % 50 == 0:
            print('epoch pixelacc: ', pixelacc/numimgs)
            #print('epoch mean_iou: ', np.mean(intersect/union))
            print('epoch mean_iou: ', np.mean(intersect[union != 0]/union[union != 0]))
            print('union[union != 0].shape: ', union[union != 0].shape)
            end = time()
            print('total time: ', end - start)
            if i == 0:
                break

        training_loss = total_loss/numimgs
        training_loss_list.append(training_loss)
        epoch_list = range(len(training_loss_list))
        mean_iou = np.mean(intersect/union)
        pixelacc = pixelacc/numimgs
        print('training pixelacc: ', pixelacc)
        print('training mean_iou: ', mean_iou)
        print('training intersect/union: ', intersect/union)
        print('training loss: ', training_loss)
        pixelacclist.append(pixelacc)
        mean_ioulist.append(mean_iou)

        with open(resultsdir + '/training_loss_list.pkl', 'wb') as f:
            pickle.dump(training_loss_list, f)
        with open(resultsdir + '/pixelacclist.pkl', 'wb') as f:
            pickle.dump(pixelacclist, f)
        with open(resultsdir + '/mean_ioulist.pkl', 'wb') as f:
            pickle.dump(mean_ioulist, f)


        if compute_val_over_whole_data:
            model.train(False)
            print('model.training: ', model.training)
            model.eval()
            print('model.training: ', model.training)
            pixelacc_val, meaniou_val, val_loss, intersect_val, union_val = compute_accuracy(val_data_loader, dataset_name='kittiCamVid', imgdir=resultsdir + '/imgs', criterion=criterion, model=model, modelname='UperNet', gt_present=True, save_images=False, epoch=e, num_classes=model.num_classes) 
            intersect_union_val = intersect_val/union_val
            print('pixelacc_val: ', pixelacc_val)
            print('meaniou_val: ', meaniou_val)
            print('val_loss: ', val_loss)
            print('intersect_val/union_val: ', intersect_union_val)
            pixelacc_val_list.append(pixelacc_val)
            meaniou_val_list.append(meaniou_val)
            loss_val_list.append(val_loss)

        with open(resultsdir + '/pixelacc_val_list.pkl', 'wb') as f:
            pickle.dump(pixelacc_val_list, f)
        with open(resultsdir + '/meaniou_val_list.pkl', 'wb') as f:
            pickle.dump(meaniou_val_list, f)
        with open(resultsdir + '/loss_val_list.pkl', 'wb') as f:
            pickle.dump(loss_val_list, f)

        if training_loss <= best_loss:
            #path = 'results/trial0/bestlosssegnetmodelnew.pt'
            path = resultsdir + '/bestlosssegnetmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones, mean_iou)
            best_loss = training_loss

        if best_loss != training_loss:
            #path = 'results/trial0/latestsegnetmodelnew.pt'
            path = resultsdir + '/latestsegnetmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones, mean_iou)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = resultsdir + '/bestvallosssegnetmodelnew.pt'
            save_model(val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, meaniou_val)

        if meaniou_val > best_mean_iou:
            best_mean_iou = meaniou_val
            path = resultsdir + '/bestmeaniousegmentmodelnew.pt'
            save_model(val_loss, path, e, model, optimizer, lr_schedule, lr_milestones, meaniou_val)
        #plt.plot(epoch_list, training_loss_list)
        #plt.xlabel('epochs')
        #plt.ylabel('training loss')
        # giving a title to my graph
        #plt.title('training loss vs epochs')
        #plt.show()
