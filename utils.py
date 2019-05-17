import  numpy as np
import  cv2
import matplotlib.pyplot as plt
import  torch
from PIL import  Image
from collections import OrderedDict
from siamrpn import  *
def crop_and_pad(img, cx, cy, model_sz, original_sz, img_mean=None):
    im_h, im_w, _ = img.shape

    xmin = cx - (original_sz - 1) / 2
    xmax = xmin + original_sz - 1
    ymin = cy - (original_sz - 1) / 2
    ymax = ymin + original_sz - 1

    left = int(round(max(0., -xmin)))
    top = int(round(max(0., -ymin)))
    right = int(round(max(0., xmax - im_w + 1)))
    bottom = int(round(max(0., ymax - im_h + 1)))

    xmin = int(round(xmin + left))
    xmax = int(round(xmax + left))
    ymin = int(round(ymin + top))
    ymax = int(round(ymax + top))
    r, c, k = img.shape
    #判断是否存在非零数
    if any([top, bottom, left, right]):
        te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top:top + r, left:left + c, :] = img
        if top:
            te_im[0:top, left:left + c, :] = img_mean
        if bottom:
            te_im[r + top:, left:left + c, :] = img_mean
        if left:
            te_im[:, 0:left, :] = img_mean
        if right:
            te_im[:, c + left:, :] = img_mean
        im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    else:
        im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
    """测试图片"""
    # plt.imshow(im_patch_original)
    # plt.show()
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
    else:
        im_patch = im_patch_original
    """测试图片"""#之后要转RGB图
    # plt.imshow(im_patch)
    # plt.show()
    scale = model_sz / im_patch_original.shape[0]
    return im_patch, scale


def get_instance_image(img, bbox, size_z, size_x, context_amount, img_mean=None):
    cx, cy, w, h = bbox  # float type
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)  # the width of the crop box
    scale_z = size_z / s_z

    s_x = s_z * size_x / size_z
    """测试图片"""
    # plt.imshow(img)
    # plt.show()
    instance_img, scale_x = crop_and_pad(img, cx, cy, size_x, s_x, img_mean)
    w_x = w * scale_x
    h_x = h * scale_x
    """测试图片"""
    # plt.imshow(instance_img)
    # plt.show()
    return instance_img, w_x, h_x, scale_x


"""anchor generate"""
def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # (5,4x225) to (225x5,4)
    ori = - (score_size // 2) * total_stride
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    #anchor
    # with open("anchor.txt","a") as F:
    #     F.write(str(anchor))

    return anchor

def adjust_learning_rate(optimizer,decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

"""计算交并比"""
def compute_iou(anchors, box):
    if np.array(anchors).ndim == 1:
        anchors = np.array(anchors)[None, :]
    else:
        anchors = np.array(anchors)
    if np.array(box).ndim == 1:
        box = np.array(box)[None, :]
    else:
        box = np.array(box)
    gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

    anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
    anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
    anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
    anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

    gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
    gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
    gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
    gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

    xx1 = np.max([anchor_x1, gt_x1], axis=0)
    xx2 = np.min([anchor_x2, gt_x2], axis=0)
    yy1 = np.max([anchor_y1, gt_y1], axis=0)
    yy2 = np.min([anchor_y2, gt_y2], axis=0)

    inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                           axis=0)
    area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
    area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
    return iou

def box_transform(anchors, gt_box):
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    gt_cx, gt_cy, gt_w, gt_h = gt_box

    target_x = (gt_cx - anchor_xctr) / anchor_w
    target_y = (gt_cy - anchor_yctr) / anchor_h
    target_w = np.log(gt_w / anchor_w)
    target_h = np.log(gt_h / anchor_h)
    regression_target = np.hstack((target_x, target_y, target_w, target_h))

    return regression_target

"""合并回归标签和分类标签"""
def union(label_1,label_2):
    """

    :param label_1: 最后一维为4(N,X,X)
    :param label_2: 最后一维为1(N,X)
    :return:target
    """
    label_2=label_2.reshape(label_1.shape[0],-1,1)
    label_2=label_2.cpu()
    label_1=label_1.cpu()
    target=np.append(label_2,label_1,axis=2)
    target=torch.from_numpy(target).float()
    return target

"""模板裁剪"""
def crop(image,output_sizes):
    """
    精确裁剪
    :param image: 模板图片（N，N，3）
    :param output_sizes: 模板尺寸,127
    :return: 图片
    """
    h,w,_=image.shape
    cy=(h-1)/2
    cx=(w-1)/2
    xmin=cx-(output_sizes-1)/2
    xmax=cx+(output_sizes-1)/2
    ymin=cy-(output_sizes-1)/2
    ymax=cy+(output_sizes-1)/2
    resized_image=image[int(ymin):int(ymax+1),int(xmin):int(xmax+1),:]
    return resized_image

def get_exemplar_image(img,bbox,size_z,context_amount,img_mean=None):
    cx,cy,w,h=bbox
    wc_z=w+context_amount*(w+h)
    hc_z=h+context_amount*(w+h)
    s_z=np.sqrt(wc_z*hc_z)
    scale_z=size_z/s_z
    exemplar_img,_=crop_and_pad(img,cx,cy,size_z,s_z,img_mean)
    return  exemplar_img,scale_z,s_z

def box_transform_inv(anchors, offset):
    anchor_xctr = anchors[:, :1]
    anchor_yctr = anchors[:, 1:2]
    anchor_w = anchors[:, 2:3]
    anchor_h = anchors[:, 3:]
    offset_x, offset_y, offset_w, offset_h = offset[:, :1], offset[:, 1:2], offset[:, 2:3], offset[:, 3:],

    box_cx = anchor_w * offset_x + anchor_xctr
    box_cy = anchor_h * offset_y + anchor_yctr
    box_w = anchor_w * np.exp(offset_w)
    box_h = anchor_h * np.exp(offset_h)
    box = np.hstack([box_cx, box_cy, box_w, box_h])
    return box

def freeze_layers(model):
    print('------------------------------------------------------------------------------------------------')
    for layer in model.featureExtract[:10]:
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
            for k, v in layer.named_parameters():
                v.requires_grad = False
        elif isinstance(layer, nn.Conv2d):
            for k, v in layer.named_parameters():
                v.requires_grad = False
        elif isinstance(layer, nn.MaxPool2d):
            continue
        elif isinstance(layer, nn.ReLU):
            continue
        else:
            raise KeyError('error in fixing former 3 layers')
    print("fixed layers:")
    print(model.featureExtract[:10])



if __name__=="__main__":
    pass
    # #test module
    # img_dir = "F:/Python_proj/data/vid15rpn_finetune/ILSVRC2015_train_00000000/000000.00.x_102.18_29.90.jpg"
    # image = Image.open(img_dir)
    # image = np.asarray(image)
    # image=crop(image,127)
    # path="F:\\Python_proj\\python_track\\alexnet.pth"
    # model_dict=load_model(path)
    # model=SiameseAlexNet()
    # model.load_state_dict(model_dict,strict=False)
    #
