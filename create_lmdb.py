import lmdb
import cv2
import numpy as np
import os
import hashlib
import functools

from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed
import multiprocessing as mp
from config import config

def worker(video_name):
    image_names = glob(video_name + '/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        try:
            _, img_encode = cv2.imencode('.jpg', img)
        except:
            embed()
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv


def create_lmdb(data_dir="F:/Python_proj/data/vid15rpn_finetune",
                output_dir="F:/Python_proj/data/vid15rpn_finetune.lmdb",
                num_threads=mp.cpu_count()):
    if config.ubuntu:
        data_dir="/2TB/vid15rpn_finetune"
        output_dir="/2TB/vid15rpn_finetune.lmdb"
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if 'meta_data.pkl' not in x]#list str
    #video_names = [x for x in video_names if os.path.isdir(x)]#list,str
    """ubuntu改大小"""
    if config.ubuntu:
        db = lmdb.open(output_dir, map_size=int(50e9))
    else:
        db = lmdb.open(output_dir, map_size=int(5e8))
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)


if __name__ == '__main__':
   Fire(create_lmdb)
