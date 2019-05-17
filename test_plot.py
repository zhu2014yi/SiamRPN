import argparse
import os
from glob import  glob
import numpy as np
import re
import json
#import setproctitle
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
    with open("result.json", 'r') as load_f:
        results = json.load(load_f)
    # ------------ starting evaluation  -----------
    data_path = "F:/Python_proj/data/OTB/"
    save_name="results.json"
    results_eval = {}
    for model in list(results.keys()):
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








