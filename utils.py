import os
import random

import numpy as np
import torch

COLOR_LIST = [
    [255, 255, 255], [190, 210, 255], [0, 255, 197], [38, 115, 0],
    [163, 255, 115], [76, 230, 0], [85, 255, 0], [115, 115, 0],
    [168, 168, 0], [255, 255, 0], [115, 178, 255], [0, 92, 230],
    [0, 38, 115], [122, 142, 245], [0, 168, 132], [115, 0, 0],
    [255, 127, 127], [255, 190, 190], [255, 190, 232], [255, 0, 197],
    [230, 0, 169], [168, 0, 132], [115, 0, 76], [255, 115, 223],
    [161, 161, 161]
]

COLOR_LIST_BGR = [color[::-1] for color in COLOR_LIST]


def map_to_color(img, label=None):
    img = img.astype(np.int8) + 1
    img_color = np.take(COLOR_LIST_BGR, img, axis=0, mode='clip')

    if label is not None:
        img_color[label == -1] = COLOR_LIST_BGR[0]
    else:
        img_color[img == 0] = COLOR_LIST_BGR[0]

    return img_color


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def gen_confusion_matrix(num_class, img_predict, img_label):
    mask = (img_label != -1)
    label = num_class * img_label[mask] + img_predict[mask]
    count = torch.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix


def eval_metrics(confusion_matrix, mode='ts'):
    eps = 1e-7

    unique_index = np.where(np.sum(confusion_matrix, axis=1) != 0)[0]
    confusion_matrix = confusion_matrix[unique_index, :]
    confusion_matrix = confusion_matrix[:, unique_index]

    a = np.diag(confusion_matrix)
    b = np.sum(confusion_matrix, axis=0)
    c = np.sum(confusion_matrix, axis=1)

    pa = a / (c + eps)
    ua = a / (b + eps)
    f1 = 2 * pa * ua / (pa + ua + eps)
    mean_f1 = np.nanmean(f1)

    oa = np.sum(a) / np.sum(confusion_matrix)

    pe = np.sum(b * c) / (np.sum(c) * np.sum(c))
    kappa = (oa - pe) / (1 - pe)

    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    iou = intersection / union
    mean_iou = np.nanmean(iou)

    f1 = np.round(f1, 3)
    if mode == 'ts':
        return mean_f1, oa, kappa, mean_iou, f1
    else:
        return mean_f1
