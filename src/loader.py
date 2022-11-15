import json
from typing import Callable, List, Dict, Tuple

import torch
import torch.utils.data.dataset
import torchvision
from easydict import EasyDict

from src.utils import split_data_8_2

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5),
])


class HerbariumDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, sample_list: List[dict], transforms: torchvision.transforms.Compose, *args, **kwargs) -> None:
        super(HerbariumDataset, self).__init__(root, transforms, *args, **kwargs)

        self.sample_list = sample_list
        self.loader = torchvision.datasets.folder.default_loader

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        """
        返回一个样本
        :param index: 样本下标
        :return: 图片Tensor， 图片标签
        """
        path = self.root + 'train_images/' + self.sample_list[index]["file_name"]
        image = self.loader(path)
        sample = self.transforms(image)
        return sample, self.sample_list[index]

    def __len__(self) -> int:
        return len(self.sample_list)


class HerbariumTestDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, transforms: torchvision.transforms.Compose, *args, **kwargs) -> None:
        super(HerbariumTestDataset, self).__init__(root, transforms, *args, **kwargs)
        with open(root + 'test_metadata.json') as fp:
            self.test_metadata_json: List[Dict] = json.load(fp)
        self.loader: Callable = torchvision.datasets.folder.default_loader

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        返回一个样本
        :param index: 样本下标
        :return: 图片Tensor， image_id
        """
        path = self.root + 'test_images/' + self.test_metadata_json[index]['file_name']
        image = self.loader(path)
        sample = self.transforms(image)
        return sample, int(self.test_metadata_json[index]['image_id'])

    def __len__(self) -> int:
        return len(self.test_metadata_json)


def get_dataloader(config: EasyDict) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_samples, val_samples = split_data_8_2(config.trainer.dataset_path + 'train_metadata.json')

    train_dataset = HerbariumDataset(root=config.trainer.dataset_path, sample_list=train_samples, transforms=train_transform)
    val_dataset = HerbariumDataset(root=config.trainer.dataset_path, sample_list=val_samples, transforms=test_transform)

    # train_subset = torch.utils.data.Subset(train_dataset, range(int(len(train_dataset) * 0.001)))  # TODO
    # val_subset = torch.utils.data.Subset(val_dataset, range(int(len(val_dataset) * 0.001)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.trainer.batch_size, shuffle=True)

    return train_loader, val_loader
