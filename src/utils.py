import json
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from accelerate import Accelerator


def load_model_dict(download_path, save_path=None, check_hash=True) -> OrderedDict:
    if download_path.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(download_path, map_location=torch.device('cpu'))
    return state_dict


def resume_train_state(path: str, train_loader: torch.utils.data.DataLoader, accelerator: Accelerator):
    try:
        # Get the most recent checkpoint
        base_path = os.getcwd() + '/' + path
        dirs = [base_path + '/' + f.name for f in os.scandir(base_path) if f.is_dir()]
        dirs.sort(key=os.path.getctime)  # Sorts folders by date modified, most recent checkpoint is the last
        accelerator.load_state(dirs[-1])
        training_difference = os.path.splitext(dirs[-1])[0]
        starting_epoch = int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
        step = starting_epoch * len(train_loader)
        accelerator.print(f'加载训练状态成功！从{starting_epoch}开始训练')
        return starting_epoch, step
    except Exception as e:
        accelerator.print(e)
        accelerator.print(f'加载训练状态失败！')
        return 0, 0


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


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
