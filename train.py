import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb
import torch.nn as nn
import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict
from siamrpn import *
from dataset import *
from utils import *
from loss import  *
if __name__=="__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    if config.ubuntu:
        torch.cuda.manual_seed(1)
        # cudaI加速
        torch.backends.cudnn.benchmark = True

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    meta_data = []
    train_videos = []
    valid_videos = []
    for i in range(len(config.datasets)):
        meta_data_path = os.path.join(config.data_dir[i], "meta_data.pkl")
        meta_data.append(pickle.load(open(meta_data_path, 'rb')))  # str

        all_videos = [x[0] for x in meta_data[i]]  # list,str

        train_videos.append(train_test_split(all_videos,
                                             test_size=1 - config.train_ratio,
                                             random_state=config.seed)[0])  # list,str
        valid_videos.append(train_test_split(all_videos,
                                             test_size=1 - config.train_ratio,
                                             random_state=config.seed)[1])  # list,str
    #open lmdb
    db = []
    if config.ubuntu:
        for i in range(len(config.datasets)):
            db.append(lmdb.open(config.data_dir[i] + '.lmdb', readonly=True, map_size=int(50e9)))
    else:
        for i in range(len(config.datasets)):
            db.append(lmdb.open(config.data_dir[i] + '.lmdb', readonly=True, map_size=int(5e8)))

    train_dataset = ImagnetVIDDataset(config.datasets, db, train_videos,
                                      config.data_dir)
    valid_dataset = ImagnetVIDDataset(config.datasets, db, valid_videos,
                                      config.data_dir
                                      , training=False)

    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True,
                             num_workers=config.valid_num_workers, drop_last=True)

    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    """开始训练"""
    #初始化网络和权重
    model = SiameseAlexNet()
    model.init_weights()
    #冻结前三层，用预训练模型
    if config.pretrained_model_path:
        checkpoint=torch.load(config.pretrained_model_path)
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint



    model=model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=config.lr,
                              momentum=config.momentum,weight_decay=config.weight_decay)
    start_epoch=1
    if config.model_path and config.init:
        print("init checkpoint %s" % config.model_path)
        checkpoint = torch.load(config.model_path)
        if "model" in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("inited checkpoint")
    if config.model_path and not config.init:
        print("loading checkpoint %s" % config.model_path)
        checkpoint = torch.load(config.model_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #el checkpoint
        del checkpoint
        torch.cuda.empty_cache()
        print("loaded checkpoint")
    #分布式
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        # 冻结前三层
    if config.fix_former_3_layers:
        if torch.cuda.device_count() > 1:
            freeze_layers(model.module)
        else:
            freeze_layers(model)

    for epoch in range(start_epoch,config.epoch+1):
        # if config.fix_former_3_layers:
        #     if torch.cuda.device_count() > 1:
        #         freeze_layers(model.module)
        #     else:
        #         freeze_layers(model)
        train_loss = []
        model.train()
        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(trainloader):
            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            labels=union(regression_target,conf_target)
            exemplar_imgs=exemplar_imgs.to(device)
            instance_imgs=instance_imgs.to(device)
            labels=labels.to(device)# float32,(8,1445,5)
            pred_score, pred_regression=model(exemplar_imgs,instance_imgs)
            closs,rloss,tloss=Multiloss(pred_score,pred_regression,labels,config.lambd)
            optimizer.zero_grad()
            tloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/cls_loss', closs.data, step)
            summary_writer.add_scalar('train/reg_loss', rloss.data, step)
            train_loss.append(tloss.detach().cpu())
            loss_temp_cls += closs.detach().cpu().numpy()
            loss_temp_reg += rloss.detach().cpu().numpy()
            if (i + 1) % config.show_interval == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
        train_loss = np.mean(train_loss)
        """验证集"""
        valid_loss = []
        model.eval()
        for i, data in enumerate(validloader):
            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            labels = union(regression_target, conf_target)
            exemplar_imgs = exemplar_imgs.to(device)
            instance_imgs = instance_imgs.to(device)
            labels = labels.to(device)  # float32
            pred_score, pred_regression = model(exemplar_imgs, instance_imgs)
            closs, rloss, tloss = Multiloss(pred_score, pred_regression, labels, config.lambd)
            valid_loss.append(tloss.detach().cpu())
        valid_loss = np.mean(valid_loss)
        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        summary_writer.add_scalar('valid/loss',
                                      valid_loss, (epoch + 1) * len(trainloader))
        adjust_learning_rate(optimizer,
                             config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
        if  epoch >4:
            save_name = "./models/siamrpn_{}.pth".format(epoch)
            new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k, v in model.state_dict().items():
                    namekey = k[7:]  # remove `module.`
                    new_state_dict[namekey] = v
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            #print('save model: {}'.format(save_name))



