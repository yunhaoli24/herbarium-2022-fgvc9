import pandas as pd
import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from torchvision.models import vit_b_16
from tqdm.auto import tqdm

from src.loader import HerbariumTestDataset, target_transform


def kaggle_test(model: nn.Module, config: EasyDict):
    accelerator = Accelerator()

    accelerator.print(config)
    accelerator.print('加载数据集...')
    test_dataset = HerbariumTestDataset(root=config.trainer.dataset_path, transforms=target_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.trainer.batch_size, shuffle=False)
    test_loader, model = accelerator.prepare(test_loader, model)

    ids = []
    predictions = []
    for images, image_ids in tqdm(test_loader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            logits = model(images)
        predictions.extend(list(logits.argmax(dim=-1).detach().cpu().numpy()))
        ids.extend(list(image_ids))
    if accelerator.is_local_main_process:
        submission = pd.DataFrame({"id": ids, "Predicted": predictions}).set_index("id")
        submission.sort_index()
        submission.to_csv("submission.csv")


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    model: nn.Module = vit_b_16(pretrained=False, num_classes=config.model.num_classes)
    model.load_state_dict(torch.load(config.model.save_name, map_location=torch.device('cpu')))

    kaggle_test(model, config)
