import argparse
import os
from glob import  glob
import numpy as np
import re
import json
import functools
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from run_SiamRPN import *


def embed_numbers(s):
    re_digit=re.compile(r"(\d+)")
    pieces=re_digit.split(s)
    return int(pieces[1])



def embeded_numbers_results(s):
    re_digits=re.compile(r"(\d+)")
    pieces=re_digits.split(s)
    return int(pieces[-2])

def cal_iou(box1,box2):
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou

def cal_success(iou):
    success_all=[]
    overlap_thresholds=np.arange(0,1.05,0.05)
    for overlap_threshold in overlap_thresholds:
        success=sum(np.array(iou)>overlap_threshold)/len(iou)
        success_all.append(success)
    return np.array(success_all)


if __name__=="__main__":
    if config.ubuntu:
        data_path="/ssd/OTB100/"
    else:
        data_path="F:/Python_proj/data/OTB/"
    video_paths=sorted(glob(data_path+"*"))
    video_paths=sorted( [video_path for video_path in video_paths if os.path.isdir(video_path)])
    if config.ubuntu:
        video_names = sorted([x.split("/")[-1] for x in video_paths])
    else:
        video_names = sorted([x.split("\\")[-1] for x in video_paths])
    # ------------ prepare models  -----------
    if config.ubuntu:
        input_paths = "/home/zhuyi/Code/SiamRPN/models"
    else:
        input_paths="F:/Python_proj/python_track/SiamRPN/models"
    model_paths = glob(input_paths + "/*.pth")
    #model_paths=sorted(x.split("\\")[-1] for x in model_paths)
    # ------------ starting validation  -----------
    results = {}
    for model_path in tqdm(model_paths, total=len(model_paths)):
        results[os.path.abspath(model_path)] = {}
        if config.ubuntu:
            model_name = os.path.abspath(model_path).split("/")[-1]
        else:
            model_name=os.path.abspath(model_path).split("\\")[-1]
        if not os.path.exists(model_name):
            os.mkdir(model_name)

        for video_path in tqdm(video_paths, total=len(video_paths)):
            groundtruth_path = video_path + '/groundtruth_rect.txt'
            """读取用flaot"""
            with open(groundtruth_path, 'r') as f:
                boxes = f.readlines()
            if ',' in boxes[0]:
                boxes = [list(map(int, box.split(','))) for box in boxes]
            else:
                boxes = [list(map(int, box.split())) for box in boxes]
            boxes = [np.array(box) - [1, 1, 0, 0] for box in boxes]
            result = run_SiamRPN(video_path, model_path, boxes[0])
            result_boxes = [np.array(box) + [1, 1, 0, 0] for box in result['res']]#list  725行4列 ndarray每一行tuple，（4，），float64
            """ubuntu 改"""
            if config.ubuntu:
                results[os.path.abspath(model_path)][video_path.split('/')[-1]] = [box.tolist() for box in
                                                                                    result_boxes]
            else:
                results[os.path.abspath(model_path)][video_path.split('\\')[-1]] = [box.tolist() for box in result_boxes]
            #保存文件
            if config.ubuntu:
                name = video_path.split('/')[-1]
            else:
                name = video_path.split('\\')[-1]
            sub_path=model_path.split("models")[0]
            sub_path = os.path.join(sub_path,model_name,name)
            current_video_path=sub_path+".txt"
            current_video_box_results=[box.tolist() for box in result_boxes]

            np.savetxt(current_video_path , current_video_box_results, fmt="%.3f", delimiter=",")

    #字典键值: 'siamrpn_28.pth_Basketball' 用split就能分开
    json.dump(results, open("result.json", 'w'))

    # ------------ starting evaluation  -----------
    if  config.ubuntu:
        data_path="/ssd/OTB100/"
    else:
        data_path = "F:/Python_proj/data/OTB/"
    save_name="results.json"
    results_eval = {}
    for model in sorted(list(results.keys()), key=embeded_numbers_results):
        results_eval[model] = {}
        success_all_video = []
        for video in results[model].keys():
            result_boxes = results[model][video]
            with open(data_path + video + '/groundtruth_rect.txt', 'r') as f:
                result_boxes_gt = f.readlines()
            if ',' in result_boxes_gt[0]:
                result_boxes_gt = [list(map(int, box.split(','))) for box in result_boxes_gt]
            else:
                result_boxes_gt = [list(map(int, box.split())) for box in result_boxes_gt]
            result_boxes_gt = [np.array(box) for box in result_boxes_gt]
            iou = list(map(cal_iou, result_boxes, result_boxes_gt))
            success = cal_success(iou)
            auc = np.mean(success)
            success_all_video.append(success)
            results_eval[model][video] = auc
        results_eval[model]['all_video'] = np.mean(success_all_video)
        print(model.split('/')[-1] + ' : ', np.mean(success_all_video))
    json.dump(results_eval, open('eval_' + save_name, 'w'))








