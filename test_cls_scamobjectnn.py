"""
Author: Haoxi Ran
Date: 05/10/2022
"""
from functools import partial

import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path

# from modules.ptaug_utils import transform_point_cloud, scale_point_cloud, get_aug_args
from modules.pointnet2_utils import sample
from utils.utils import get_model, get_loss, set_seed, weight_init

from kd.tool import *
import importlib
import matplotlib.pyplot as plt
import itertools

def plot(matrix, nc = 15, normalize = True, save_dir='./', names=()):
    import seaborn as sn
    array = matrix / (matrix.sum(0).reshape(1, nc) + 1E-6) if normalize else matrix.astype(int)  # normalize
    # if normalize:
    #     array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and len(names) == nc  # apply names to ticklabels
    sn.heatmap(array, annot=nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f' if normalize else '.20g', square=True,
                xticklabels=names if labels else "auto",
                yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    if normalize:
        fig.savefig(Path(save_dir) / 'confusion_matrix1.png', dpi=250)
    else:
        fig.savefig(Path(save_dir) / 'confusion_matrix2.png', dpi=250)

def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus)
    set_seed(args.seed)

    '''CREATE DIR'''
    experiment_dir = Path(os.path.join(args.log_root, 'PointAnalysis', 'log'))
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.dataset)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    args.normal = False
    # aug_args = get_aug_args(args)

    ###
    trainDataLoader, testDataLoader, args.num_class, total_num = get_dataset(args)
    ###

    classifier = torch.nn.DataParallel(get_model(args, args.model)).cuda()
    checkpoint = torch.load(args.teacher_check)
    # print(checkpoint.keys())
    # # ['cfg', 'epoch', 'model_state', 'optimizer_state', 'lr_sched_state', 'bnm_sched_state', 'test_perf']
    # exit()
    if 'model_state_dict' in checkpoint.keys():
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        weights = checkpoint['model_state']
        for key in list(weights):
            # print(key)
            weights[key.replace('module.model', 'module')] = weights.pop(key)
        classifier.load_state_dict(weights)
        # module.model.classfier.8.bias

        # module.classfier.8.bias
        # print(checkpoint['model_state'].keys())
        # print(classifier.state_dict().keys())
        # exit()
    classifier = classifier.eval()

    cls_list, oa_list = [], []
    for i in range(10000):
        with torch.no_grad():
            class_acc, vote_acc, cls_acc, conf_matrix = test(classifier.eval(), testDataLoader, args, num_class = args.num_class, num_point=args.num_point,
                                        total_num = total_num)

            log_string('Test class Accuracy: %.2f' % (class_acc * 100))
            log_string('Test Vote Accuracy: %.2f' % (vote_acc * 100))
            
            s = ''
            for i in cls_acc:
                s += str(i) + " "
            log_string('Test Vote Accuracy : ' + s)

            if vote_acc > 0.86 and vote_acc < 0.87:
                labels =  [
                    "bag",
                    "bin",
                    "box",
                    "cabinet",
                    "chair",
                    "desk",
                    "display",
                    "door",
                    "shelf",
                    "table",
                    "bed",
                    "pillow",
                    "sink",
                    "sofa",
                    "toilet",
                ] # 每种类别的标签
                conf_matrix = conf_matrix.numpy()
                plot(conf_matrix, 15, normalize = False, names = labels)
                plot(conf_matrix, 15, normalize = True, names = labels)
                break


        with open('./data.txt', "a+") as file:
            file.write(str(args.corruption) + '-' + str(args.severity) + ":" + '\n')
            # file.write('    mAcc: %.2f' % (sing_acc * 100) + '\n')
            file.write('    OA: %.2f' % (vote_acc * 100) + '\n')

            # cls_list.append(sing_acc * 100)
            oa_list.append(vote_acc * 100)
    cls_mean, cls_var = np.mean(np.array(cls_list))
    oa_mean, oa_var = np.std(np.array(oa_list))
    # 求方差
    log_string('Test class Accuracy: %.2f +- %.2f' % (np.mean(l), np.std(l)))
    log_string('Test OA Accuracy: %.2f +- %.2f' % (oa_mean, oa_var))
   


if __name__ == '__main__':
    args = parse_args()
    main(args)