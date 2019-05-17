import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET
from torch.utils.data.dataset import Dataset
from config import config
from utils import *
import torchvision.transforms as transforms
class ImagnetVIDDataset(Dataset):
    def __init__(self,dataset_names,db, video_names, data_dir, training=True):
        super(ImagnetVIDDataset, self).__init__()
        self.dataset_names = dataset_names
        self.video_names=video_names
        self.data_dir = data_dir
        # data augmentation
        self.max_stretch = config.scale_resize
        self.max_translate = config.max_translate
        self.random_crop_size = config.instance_size
        self.training = training
        if self.training:
            self.z_transforms=transforms.Compose([
                #RandomStretch(config.scale_resize),
                #CenterCrop((config.exemplar_size, config.exemplar_size)),
                ToTensor()])
            self.x_transforms=transforms.Compose([ToTensor()])
        #验证集
        else:
            self.z_transforms = transforms.Compose([
                            #CenterCrop((config.exemplar_size,config.exemplar_size)),
                            ToTensor()])
            self.x_transforms = transforms.Compose([
        ToTensor()
    ])
        self.meta_data = []
        self.txn = []
        # vid + got
        for i, data_path in enumerate(data_dir):
            meta_data_path = os.path.join(data_path, 'meta_data.pkl')  # str
            meta_data = pickle.load(open(meta_data_path, 'rb'))  # list,tuple(0,1),0:视频名字,1:视频图片数目
            self.meta_data.append({x[0]: x[1] for x in meta_data})  # 同上
            # filter traj len less than 过滤少于2帧的视频
            if data_path.split("/")[-1].split("_")[-2]=="coco":
                continue
            else:
                for key in self.meta_data[i].keys():
                    trajs = self.meta_data[i][key]
                    for trkid in list(trajs.keys()):
                        if len(trajs[trkid]) < 2:
                            del trajs[trkid]

        for i, db in enumerate(db):
            self.txn.append(db.begin(write=False))

        self.num = len(self.video_names) if config.num_per_epoch is None or not training \
            else config.num_per_epoch

        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        config.valid_scope)
    ##读取图片，通道为BGR
    def imread(self,path,txn):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img
        #index 随机采样得来
    def __getitem__(self, idx):
        # num_neg=0
        # num_pos=0
        # while ((num_pos<4)&(num_neg<47)):
        while (True):
            #while True:

            index_dataset = self.dataset_names  # choice vid or got-10k or coco or LaSOT
            index_dataset = np.random.choice(len(index_dataset))

            data_name=self.dataset_names[index_dataset]

            idx = idx % len(self.video_names[index_dataset])
            video = self.video_names[index_dataset][idx]
            trajs = self.meta_data[index_dataset][video]
            data_dir = self.data_dir[index_dataset]
            txn = self.txn[index_dataset]

            if (len(trajs.keys())==0) :
                idx=np.random.randint(self.num)
                #assert len(trajs.keys())==0,"Error trajs is zero!"
                continue
            else:
                pass

            if data_name=="COCO":
                video_path=os.path.join(data_dir,trajs[0][0].split(".")[-2]+"/*")
                pairs_name=glob.glob(video_path)
                if len(pairs_name)==1:
                    exemplar_name,instance_name=np.random.choice(pairs_name,size=2)
                else:
                    exemplar_name,instance_name=np.random.choice(pairs_name,size=2,replace=False)

            else:

                trkid = np.random.choice(list(trajs.keys()))
                traj = trajs[trkid]
                assert len(traj) > 1, "video_name: {}".format(video)
                # sample exemplar
                exemplar_idx = np.random.choice(list(range(len(traj))))
                exemplar_name = os.path.join(data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))
                if data_name=="LaSOT":
                    img_idx=glob.glob(exemplar_name)
                    if len(img_idx)==0:
                        idx=np.random.randint(self.num)
                        continue
                    else:
                        exemplar_name=img_idx[0]
                else:
                    exemplar_name = \
                        glob.glob(os.path.join(data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
                # with open("debug.txt","r") as f:
                #     f.write(exemplar_name)

            exemplar_img = self.imread(exemplar_name,txn)
            #exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)# ndarray ,uint8 ,321,321,3
            exemplar_img=crop(exemplar_img,config.exemplar_size)
            exemplar_img = self.z_transforms(exemplar_img)


            if data_name=="COCO":
                pass
            else:# sample instance
                low_idx = max(0, exemplar_idx - config.frame_range)
                up_idx = min(len(traj), exemplar_idx + config.frame_range)
                # create sample weight, if the sample are far away from center
                # the probability being choosen are high
                #以自己为中心
                weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)
                instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)#以自己为中心采样(array的拼接)
                # if data_name=="LaSOT":
                #     if instance in traj:
                #         pass
                #     else:
                #         idx=np.random.randint(self.num)
                #         print("init!")
                #         continue
                if data_name=="LaSOT":
                    instance_idx=glob.glob(os.path.join(data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))
                    if len(instance_idx)==0:
                        idx=np.random.randint(self.num)
                        continue
                    else:
                        instance_name=instance_idx[0]
                else:
                    instance_name = glob.glob(os.path.join(data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
            instance_img = self.imread(instance_name,txn)
            #instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)#,321,321,3 BGR转RGB
            gt_w, gt_h = float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])
            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, gt_w, gt_h)
            instance_img, gt_cx, gt_cy = self.RandomCrop(instance_img)
            if gt_h<=0 or gt_w<=0:
                idx=np.random.randint(self.num)
                continue

            #测试
            # plt.imshow(instance_img)
            # plt.show()
            instance_img = self.x_transforms(instance_img)
            regression_target, conf_target ,num_pos,num_neg= self.compute_target_choose(self.anchors, np.array([gt_cx, gt_cy, gt_w, gt_h]))
            if ((num_pos>=1)and(num_neg>=47)):

                break
            else:
                idx = np.random.randint(self.num)
                #print("num_pos:{},num_neg:{}\n".format(num_pos, num_neg))

        return  exemplar_img,instance_img,regression_target, conf_target
    def RandomCrop(self, sample ):
        #测试完，确定是对的
        # plt.imshow(sample)
        # plt.show()
        shape = sample.shape[:2]
        cy_o = (shape[0] - 1) // 2
        cx_o = (shape[1] - 1) // 2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        assert abs(cy - cy_o) <= self.max_translate and \
               abs(cx - cx_o) <= self.max_translate
        gt_cx = cx_o - cx
        gt_cy = cy_o - cy

        ymin = cy - self.random_crop_size // 2
        xmin = cx - self.random_crop_size // 2
        ymax = cy + self.random_crop_size // 2 + self.random_crop_size % 2
        xmax = cx + self.random_crop_size // 2 + self.random_crop_size % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch, gt_cx, gt_cy

    def compute_target_choose(self, anchors, box):
        regression_target = box_transform(anchors, box)
        iou = compute_iou(anchors, box).flatten()
        # print(np.max(iou))

        pos_index = list(np.where(iou > config.pos_threshold)[0])
        neg_index = list(np.where(iou < config.neg_threshold)[0])
        num_neg=len(neg_index)#pos index len为1可能是空或者就一个index
        #assert num_pos>0,"Error:num_pos is {}".format(num_pos)
        num_pos=len(pos_index)
        label = np.ones_like(iou) * -1
        if num_pos == 0:
            pass
        else:
            pos_index = np.random.choice(pos_index, size=min(config.num_pos, num_pos),
                                         replace=False)
            label[pos_index] = 1
        if num_neg == 0:
            pass
        else:
            neg_index = np.random.choice(neg_index, size=max(config.total_num - num_pos, config.total_num-config.num_pos),
                                         replace=False)
            label[neg_index] = 0

        return regression_target, label,num_pos,num_neg

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)


    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = int(w * scale_w) / w
        scale_h = int(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def __len__(self):
        return self.num



class RandomStretch(object):
    """
            Args:
                sample(numpy array): 3 or 1 dim image
            """
    def __init__(self,max_stretch=0.05):
        super(RandomStretch,self).__init__()
        self.max_stretch=max_stretch
    def __call__(self, sample):
        scale_h=1.0+np.random.uniform(-self.max_stretch,self.max_stretch)
        scale_w=1.0+np.random.uniform(-self.max_stretch,self.max_stretch)
        h,w=sample.shape[:2]
        shape=(int(h*scale_h)),(int(w*scale_w))
        return  cv2.resize(sample,shape,cv2.INTER_LINEAR)

class CenterCrop(object):
    """Crop the image in the center according the given size
                if size greater than image size, zero padding will adpot
            Args:
                size (tuple): desired size
            """
    def __init__(self,size):
        super(CenterCrop,self).__init__()
        self.size=size

    def __call__(self, sample):
        """
                Args:
                    sample(numpy array): 3 or 1 dim image
                """
        shape=sample.shape[:2]
        cy,cx=(shape[0]-1)//2,(shape[1]-1)//2
        ymin, xmin = cy - self.size[0] // 2, cx - self.size[1] // 2
        ymax, xmax = cy + self.size[0] // 2 + self.size[0] % 2, \
                     cx + self.size[1] // 2 + self.size[1] % 2

        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch


class RandomCrop(object):
    def __init__(self,size,max_translate):
        """Crop the image in the center according the given size
                    if size greater than image size, zero padding will adpot
                Args:
                    size (tuple): desired size
                    max_translate: max translate of random shift
                """
        self.size=size
        self.max_translate=max_translate
    def __call__(self, sample):
        """
                Args:
                    sample(numpy array): 3 or 1 dim image
                """
        shape=sample[:2]
        cy_o=(shape[0]-1)//2
        cx_o=(shape[1]-1)//2
        cy = np.random.randint(cy_o - self.max_translate,
                               cy_o + self.max_translate + 1)
        cx = np.random.randint(cx_o - self.max_translate,
                               cx_o + self.max_translate + 1)
        assert abs(cy - cy_o) <= self.max_translate and \
               abs(cx - cx_o) <= self.max_translate
        ymin = cy - self.size[0] // 2
        xmin = cx - self.size[1] // 2
        ymax = cy + self.size[0] // 2 + self.size[0] % 2
        xmax = cx + self.size[1] // 2 + self.size[1] % 2
        left = right = top = bottom = 0
        im_h, im_w = shape
        if xmin < 0:
            left = int(abs(xmin))
        if xmax > im_w:
            right = int(xmax - im_w)
        if ymin < 0:
            top = int(abs(ymin))
        if ymax > im_h:
            bottom = int(ymax - im_h)

        xmin = int(max(0, xmin))
        xmax = int(min(im_w, xmax))
        ymin = int(max(0, ymin))
        ymax = int(min(im_h, ymax))
        im_patch = sample[ymin:ymax, xmin:xmax]
        if left != 0 or right != 0 or top != 0 or bottom != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=0)
        return im_patch

class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))

