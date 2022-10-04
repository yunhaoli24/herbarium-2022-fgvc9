import json
import torch
import torchvision
from typing import Any, Optional, Callable, List, Dict, Tuple

from torchvision import transforms

target_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224, 224)),
    transforms.Normalize(mean=0.5, std=0.5),
])



class HerbariumDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, common_list: List[dict],transforms: torchvision.transforms.Compose, *args, **kwargs) -> None:
        super(HerbariumTestDataset, self).__init__(root, transforms, *args, **kwargs)

        self.common_list = common_list
        self.loader = torchvision.datasets.folder.default_loader

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        返回一个样本
        :param index: 样本下标
        :return: 图片Tensor， image_id
        """
        path = self.root + 'test_images/' + self.common_list[index]["file_name"]
        image = self.loader(path)
        sample = self.transforms(image)
        return sample, self.common_list[index]['category_id']

    def __len__(self) -> int:
        return len(self.common_list)


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
