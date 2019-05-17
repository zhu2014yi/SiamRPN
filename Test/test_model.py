from torchvision.models import alexnet
from siamrpn import *
#from utils import load_model
import  torch
import  os
import numpy as np
import cv2
from visual import *
from tracker import  *
from PIL import  Image
import cv2
import torch.nn.functional as F
if __name__=="__main__":
    """加载预训练模型并冻结一些层数"""
    # model=SiameseAlexNet()
    # path="F:\\Python_proj\\python_track\\alexnet.pth"
    # model_dict=load_model(path)
    # model.load_state_dict(model_dict,strict=False)
    #
    #
    # list_dir=['featureExtract.0.weight', 'featureExtract.0.bias',
    #    'featureExtract.1.weight', 'featureExtract.1.bias',
    #    'featureExtract.1.running_mean', 'featureExtract.1.running_var',
    #     'featureExtract.4.weight',
    #    'featureExtract.4.bias', 'featureExtract.5.weight',
    #    'featureExtract.5.bias', 'featureExtract.5.running_mean',
    #    'featureExtract.5.running_var',
    #    'featureExtract.8.weight', 'featureExtract.8.bias',
    #    'featureExtract.9.weight', 'featureExtract.9.bias',
    #    'featureExtract.9.running_mean', 'featureExtract.9.running_var',]
    # for name in model.named_parameters():
    #     if name[0] in list_dir:
    #         name[1].requires_grad = False
    """response_map"""
    # path="F:\\Python_proj\\python_track\\SiamRPN\\models\\siamrpn_64.pth"
    # tracker=SiamRPNTracker(path)
    # video_path="F:\\Python_proj\\data\\OTB2015\\Basketball\\img\\"
    # video_names="Basketball"
    # frames = [video_path + x for x in np.sort(os.listdir(video_path))]
    # frames = [x for x in frames if '.jpg' in x]
    # init_box=np.array([198.0,214.0,34.0,81.0])
    # x, y, w, h = init_box
    # for idx, frame in enumerate(frames):
    #     frame = cv2.imread(frame)#ndarray
    #     if idx == 0:
    #         tracker.init(frame, init_box)
    #         bbox = (x + w / 2 - 1 / 2, y + h / 2 - 1 / 2, w, h)
    #         bbox = np.array(bbox).astype(np.float64)
    #         if config.visual:
    #             box=np.array([x,y,w,h])
    #             show_frame(frame,box)
    #     else:
    #         bbox, score = tracker.updata(frame)  # x,y,w,h
    #         bbox = np.array(bbox)
    #         if config.visual:
    #             box=np.array([bbox[0]-bbox[2]/2+1/2,bbox[1]-bbox[3]/2+1/2,bbox[2],bbox[3]])
    #             show_frame(frame, box)
    """使用train输出map"""
#     path="F:\\Python_proj\\python_track\\SiamRPN\\models\\siamrpn_43.pth"
#     model=SiameseAlexNet()
#     checkpoint=torch.load(path,map_location="cpu")
#     model.load_state_dict(checkpoint["model"])
#     exemplar_image_path="F:\\Python_proj\\data\\OTB2015\\Basketball\\img\\0001.jpg"
#     exemplar_image=cv2.imread(exemplar_image_path)
#     exemplar_image,_,_=get_exemplar_image(exemplar_image,np.array([215.0,254.5,34.0,81.0]),size_z=127,context_amount=0.5)
#     exemplar_image=np.asarray(exemplar_image)
#     plt.imshow(exemplar_image)
#     plt.show()
#     exemplar_image=exemplar_image.transpose(2,0,1)
#     exemplar_image=exemplar_image[None,:,:,:]
#     exemplar_image=torch.from_numpy(exemplar_image.astype(np.float32))
#     instance_image_path="F:\\Python_proj\\data\\OTB2015\\Basketball\\img\\0016.jpg"
#     instance_image=cv2.imread(instance_image_path)
#     instance_image,_,_,_=get_instance_image(instance_image,np.array([
# 204.0,225.0,34.0,81.0]),127,255,0.5,)
#     instance_image = np.asarray(instance_image)
#     plt.imshow(instance_image)
#     plt.show()
#     instance_image = instance_image.transpose(2, 0, 1)
#     instance_image = instance_image[None, :, :, :]
#     instance_image = torch.from_numpy(instance_image.astype(np.float32))
#     pred,_=model(exemplar_image,instance_image)
#     pred = pred.reshape(-1, 2, 5*17*17).permute(0, 2, 1)
#     score_pred = F.softmax(pred, dim=2)[0, :, 1].cpu().detach().numpy()  ##取预测值为1的列（列属性为概率）
#
#     response_map = []
#     score_pre = score_pred.reshape(5, 289)
#     for i in range(5):
#         response_map.append(score_pre[i, :].reshape(17, 17))
#         plt.imshow(score_pre[i, :].reshape(17, 17))
#         plt.show()



    # def out_put_feature_map(model_path,video_path,gt_path):
    #     """
    #
    #     :param model_path:"F:\\Python_proj\\python_track\\SiamRPN\\models\\siamrpn_43.pth"
    #     :param video_path: F:/Python_proj/data/OTB2015/Basketball/img/
    #     :param init_box: np.array()
    #     :param gt_path: F:\Python_proj\data\OTB2015\Basketball\label/groundtruth_rect.txt
    #     :return:feature map
    #     """
    #     model = SiameseAlexNet()
    #     Totensor=ToTensor()
    #     checkpoint = torch.load(model_path, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])
    #     frames = [video_path + x for x in np.sort(os.listdir(video_path))]
    #     frames = [x for x in frames if '.jpg' in x]
    #     hann_window = np.outer(
    #         np.hanning(17),
    #         np.hanning(17))
    #     with open(gt_path, 'r') as f:
    #         boxes = f.readlines()
    #     if ',' in boxes[0]:
    #         boxes = [list(map(int, box.split(','))) for box in boxes]
    #     else:
    #         boxes = [list(map(int, box.split())) for box in boxes]
    #     for i in range(len(boxes)):
    #         boxes[i]=[boxes[i][0]+boxes[i][2]/2,boxes[i][1]+boxes[i][3]/2,boxes[i][2],boxes[i][3]]
    #     x, y, w, h = boxes[0]
    #     for idx, frame in enumerate(frames):
    #         frame = cv2.imread(frame)#ndarray
    #         if idx == 0:
    #             exemplar_image, _, _ = get_exemplar_image(frame,
    #                                                       boxes[idx],
    #                                                       size_z=127, context_amount=0.5)
    #             exemplar_image = np.asarray(exemplar_image)
    #             plt.imshow(exemplar_image)
    #             plt.show()
    #             exemplar_image = exemplar_image.transpose(2, 0, 1)
    #             exemplar_image = exemplar_image[None, :, :, :]
    #             exemplar_image = torch.from_numpy(exemplar_image.astype(np.float32))
    #
    #         else:
    #             instance_image, _, _, _ = get_instance_image(frame, np.array([300.0,245.5,34.0,81.0]), 127, 255, 0.5)
    #             instance_image = np.asarray(instance_image)
    #             plt.imshow(instance_image)
    #             plt.show()
    #             instance_image = instance_image.transpose(2, 0, 1)
    #             instance_image = instance_image[None, :, :, :]
    #             instance_image = torch.from_numpy(instance_image.astype(np.float32))
    #
    #             pred,_=model(exemplar_image,instance_image)
    #             # pred=pred.reshape(-1,10,289)
    #             # pred_0=pred[:,::2,:]
    #             # pred_1=pred[:,1::2,:]
    #             # pred=torch.cat((pred_0,pred_1),dim=1)
    #             pred = pred.reshape(-1, 2, 5 * 17 * 17).permute(0, 2, 1)
    #             score_pred = F.softmax(pred, dim=2)[0, :, 1].cpu().detach().numpy()  ##取预测值为1的列（列属性为概率）
    #
    #             response_map = []
    #             score_pre = score_pred.reshape(5, 289)
    #             for i in range(5):
    #                 response=score_pre[i, :].reshape(17, 17)
    #                 response=np.multiply(response,hann_window)
    #                 response=cv2.resize(response,(255,255))
    #                 response_map.append(response)
    #                 plt.imshow(response)
    #                 plt.show()
    #
    #
    # out_put_feature_map("F:\\Python_proj\\python_track\\SiamRPN\\models\\siamrpn_60.pth",
    #                     "F:/Python_proj/data/OTB2015/Basketball/img/",
    #                     'F:/Python_proj/data/OTB2015/Basketball/label/groundtruth_rect.txt'
    #
    #                     )
    num_pos=0
    num_neg=0
    while((num_pos>4)&(num_neg>5)):
        num_neg=np.random.randint(0,10)
        num_pos=np.random.randint(0,10)
    print("finish!")