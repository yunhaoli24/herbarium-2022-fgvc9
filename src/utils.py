import json

import numpy as np
import torch
import torch.nn as nn


def save_model(model: nn.Module, path):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def split_data_8_2(file_path):
    with open(file_path, 'rb') as dict1:
        all_dict = json.load(dict1)

    root_list = all_dict['annotations']
    file_list = all_dict['images']
    root_list.append(root_list[0])
    train_list_8 = []
    test_list_2 = []

    preid = 0
    sum = 0

    pre = 0
    end = 0

    # t=0
    # root_list
    total = 0
    for i in range(839773):
        class_id = root_list[i]["category_id"]
        if class_id != preid or i == 839772:
            # t+=1
            preid = class_id
            train_num = int(sum * 0.8)

            mid = pre + train_num
            k = pre
            total += sum

            # exit()
            while k < i:
                if k < mid:
                    root_list[k]["file_name"] = file_list[k]['file_name']
                    train_list_8.append(root_list[k])
                else:
                    root_list[k]["file_name"] = file_list[k]['file_name']
                    test_list_2.append(root_list[k])
                k += 1

            sum = 0
            pre = i
            end = i

        else:
            sum += 1
            end += 1
        if i == 839772:
            break

    return train_list_8, test_list_2


def split_mid_8(train_list):
    train_list.append(train_list[0])
    print(len)
    mid_list_1 = []
    mid_list_2 = []

    preid = 0
    sum = 0

    pre = 0
    end = 0

    total = 0
    for i in range(653139):
        class_id = train_list[i]["category_id"]
        if class_id != preid or i == 653138:
            preid = class_id
            train_num = int(sum * 0.5)

            mid = pre + train_num
            k = pre
            total += sum

            while k < i:
                if k < mid:
                    mid_list_1.append(train_list[k])
                else:
                    mid_list_2.append(train_list[k])
                k += 1

            sum = 0
            pre = i
            end = i

        else:
            sum += 1
            end += 1
        if i == 653138:
            break

    return mid_list_1, mid_list_2
