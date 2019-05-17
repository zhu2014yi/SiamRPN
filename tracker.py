import numpy as np
import  cv2
import torch
import  torch.nn.functional as F
import  time
import torchvision.transforms as transforms
from siamrpn import SiameseAlexNet
from config import config
from dataset import ToTensor
from utils import *
import torch.nn as nn
torch.set_num_threads(2) #不然会使用所有cpu
class SiamRPNTracker(object):
    def __init__(self,model_path):
        self.model=SiameseAlexNet()
        """服务器部署就改,去掉map_location"""
        if config.ubuntu:
            checkpoint = torch.load(model_path)
        else:
            checkpoint=torch.load(model_path,map_location="cpu")
        if "model" in checkpoint.keys():
            """服务器部署就改"""
            if config.ubuntu:
                self.model.load_state_dict(torch.load(model_path)["model"])
            else:
                self.model.load_state_dict(torch.load(model_path,map_location="cpu")["model"])
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0,1' if self.cuda else 'cpu')
        self.model=self.model.to(self.device)
        self.model.eval()
        self.transforms=transforms.Compose(
            [
                ToTensor()
            ]
        )
        valid_scope=config.valid_scope
        self.anchors=generate_anchors(config.total_stride,
                                      config.anchor_base_size,
                                      config.anchor_scales,
                                      config.anchor_ratios,
                                      valid_scope)
        ## np.outer()里面就是汉宁窗了,None增加了最前面的一个通道数量是1
        self.windows=np.tile(np.outer(np.hanning(config.score_size),np.hanning(config.score_size))[None,:],
                             [config.anchor_num,1,1]).flatten() #从通道0的scope_map开始展开（17,17）

    def _cosine_window(self,size):
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window
    def init(self,frame,bbox):
        self.pos=np.array([bbox[0]+bbox[2]/2-1/2,bbox[1]+bbox[3]/2-1/2])
        self.target_sz=np.array([bbox[2],bbox[3]])
        self.bbox=np.array([bbox[0]+bbox[2]/2-1/2,bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])
        self.origin_target_sz=np.array([bbox[2], bbox[3]])
        self.img_mean = np.mean(frame, axis=(0, 1)) ##先0通道均值处理再1通道均值处理
        exemplar_img,_,_=get_exemplar_image(frame,self.bbox,config.exemplar_size,config.context_amount,self.img_mean)
        #exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
        """测试图"""
        # plt.imshow(exemplar_img)
        # plt.show()
        exemplar_img=self.transforms(exemplar_img)[None,:,:,:]#None作用是增加一个维度
        self.model.track_init(exemplar_img.to(self.device))

    def updata(self,frame):
        instance_img,_,_,scale_x=get_instance_image(frame,self.bbox,config.exemplar_size,
                                                    config.instance_size,
                                                    config.context_amount,self.img_mean)
        #instance_img=cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        instance_img=self.transforms(instance_img)[None,:,:,:]#1,3,255,255
        pred_score,pred_regression=self.model.track(instance_img.to(self.device))
        # pred_score=pred_score.squeeze()
        # pred_score_0=pred_score[:,::2,:]
        # pred_score_1=pred_score[:,1::2,:]
        # pred_score=torch.cat((pred_score_0,pred_score_1),dim=1)

        pred_conf=pred_score.reshape(-1,2,config.anchor_num*config.score_size*config.score_size).permute(0,2,1)
        pred_offset=pred_regression.reshape(-1,4,config.anchor_num*config.score_size*config.score_size).permute(0,2,1)
        delta= pred_offset[0].cpu().detach().numpy()
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred=F.softmax(pred_conf,dim=2)[0,:,1].cpu().detach().numpy()##取预测值为1的列（列属性为概率）

        def change(r):
            return np.maximum(r,1./r)
        def sz(w,h):
            pad=(w+h)*0.5
            sz2=(w+pad)*(h+pad)
            return np.sqrt(sz2)
        def sz_wh(wh):
            pad=(wh[0]+wh[1])*0.5
            sz2=(wh[0]+pad)*(wh[1]+pad)
            return np.sqrt(sz2)
        s_c=change(sz(box_pred[:,2],box_pred[:,3])/(sz_wh(self.target_sz*scale_x)))  #scale penalty
        r_c=change((self.target_sz[0]/self.target_sz[1])/(box_pred[:,2]/box_pred[:,3])) # ratio penalty
        penalty=np.exp(-(r_c*s_c-1.)*config.penalty_k)
        pscore=penalty*score_pred
        pscore=pscore*(1-config.window_influence)+self.windows*config.window_influence
        best_pscore_id=np.argmax(pscore)
        #定位至原226,226,3（crop之前的图片）上坐标
        target=box_pred[best_pscore_id,:]/scale_x
        # """test_scope map"""
        # inst_image=instance_img.squeeze()
        # inst_image=inst_image.permute(1,2,0)
        # inst_image=np.asarray(inst_image,dtype=np.uint8)
        # plt.imshow(inst_image)
        # plt.show()
        # response_map=[]
        # score_pre=score_pred.reshape(5,289)
        # for i in range(5):
        #     response_map.append(score_pre[i,:].reshape(17,17))
        #     plt.imshow(score_pre[i,:].reshape(17,17))
        #     plt.show()
        #####################################
        lr=penalty[best_pscore_id]*score_pred[best_pscore_id]*config.lr_box
        res_x=np.clip(target[0]+self.pos[0],0,frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))
        return self.bbox, score_pred[best_pscore_id]