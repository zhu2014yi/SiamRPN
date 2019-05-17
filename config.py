import  numpy as np

class Config(object):
    ubuntu=True
    # 训练参数
    fix_former_3_layers = True
    datasets = ["VID",
                "GOT",
                "LaSOT",
                "COCO"
                ]
    if ubuntu:
        data_dir=["/ssd/vid15rpn_finetune",
                  "/ssd/got10k_finetune",
                  "/ssd/lasot_finetune",
                  "/ssd/coco_finetune"
                  ]
    else:
        data_dir = ["F:/Python_proj/data/vid15rpn_finetune",
                    #"F:/Python_proj/data/got10k_finetune",
                    #"F:/"
                    ]
    exemplar_size = 127  # exemplar size
    instance_size = 255  # instance size
    context_amount = 0.5  # context amount
    sample_type = 'uniform'
    frame_range = 100  # frame range of choosing the instance

    # training related
    if ubuntu:
        num_per_epoch=5*53200 #53200
    else:
        num_per_epoch = 50  # 53200 # num of samples per epoch

    train_ratio = 0.99  # training ratio of VID dataset
    frame_range = 100  # frame range of choosing the instance
    if ubuntu:
        train_batch_size=64
    else:
        train_batch_size = 8  # training batch size

    valid_batch_size = 1  # validation batch size
    if ubuntu:
        train_num_workers=6
    else:
        train_num_workers = 0  # number of workers of train dataloader
    if ubuntu:
        valid_num_workers=1
    else:
        valid_num_workers = 0  # number of workers of validation dataloader
    clip = 10
    lambd=5
    valid_scope=17   #scroe map
    start_lr = 1e-2  #之前1e-2
    end_lr = 1e-6   #之前1e-2
    warm_epoch = None
    warm_lr = 8e-4
    warm_scale = warm_lr / start_lr
    epoch = 40
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
            np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    # decay rate of LR_Schedular
    step_size = 1  # step size of LR_Schedular
    momentum = 0.9  # momentum of SGD
    weight_decay = 0.0005 # weight decay of optimizator

    seed = 1234  # seed to sample training videos
    log_dir = './models/logs'  # log dirs
    response_scale = 1e-3  # normalize of response
    max_translate = 32  # max translation of random shift
    scale_resize = 0.15  # scale step of instance image
    total_stride = 8  # total stride of backbone

    anchor_scales = np.array([8, ])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    total_num=num_pos+num_neg
    save_interval = 1
    if ubuntu:
        show_interval=20
    else:
        show_interval = 1  # 100
    #pretrained_model = None  # '/mnt/usershare/zrq/pytorch/lab/model/zhangruiqi/finaltry/sharedata/SiamRPNOTB.model'
    # pretrained_model = '/mnt/usershare/zrq/pytorch/lab/model/zhangruiqi/finaltry/sharedata/alexnet.pth'
    model_path = ""
    if ubuntu:
        pretrained_model_path="/home/zhuyi/Code/Pretrained_model/alexnet.pth"
    else:
        pretrained_model_path = "F:\\Python_proj\\python_track\\alexnet.pth"
    init = None

    # tracking related
    gray_ratio = 0.25
    blur_ratio = 0.15
    score_size = int((instance_size - exemplar_size) / 8 + 1)  #17
    penalty_k = 0.055    #0.055
    window_influence = 0.42 #0.42
    lr_box = 0.295  #0.295
    min_scale = 0.1
    max_scale = 10
    visual = False
config=Config()
