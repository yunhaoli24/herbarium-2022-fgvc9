import torch
import torch.nn as nn
import torchvision.datasets
import yaml
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from easydict import EasyDict
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from accelerate import Accelerator

from src.loader import HerbariumDataset
from src.utils import same_seeds, save_model

if __name__ == '__main__':
    # 读取配置
    accelerator = Accelerator()
    config = EasyDict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(42)
    accelerator.print(config)

    target_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    accelerator.print('加载数据集...')
    # train_dataset = HerbariumDataset(root=config.trainer.train_dataset_path, transform=target_transform)
    # val_dataset = HerbariumDataset(root=config.trainer.val_dataset_path, transform=target_transform)

    train_dataset = torchvision.datasets.CIFAR10(root="./data/", transform=target_transform, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root="./data/", transform=target_transform, train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.trainer.batch_size, shuffle=True)
    num_classes = 10

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    accelerator.print('加载模型')
    model = resnet18(pretrained=False, num_classes=num_classes)
    if config.trainer.resume:
        model.load_state_dict(torch.load(config.model.checkpoint_path, map_location=torch.device('cpu')))

    # 定义训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.lr)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.trainer.num_epochs / 10), eta_min=1e-5)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # Tensorboard
    writer = SummaryWriter()

    stale = 0
    best_acc = 0
    patience = config.trainer.num_epochs / 2
    # 开始训练
    accelerator.print("开始训练！")

    for epoch in range(config.trainer.num_epochs):
        model.train()

        # 训练
        total_loss = 0
        train_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        for features, labels in train_bar:
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_local_main_process:
                train_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training')

        # 验证
        model.eval()
        accurate = 0
        num_elements = 0
        val_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
        for features, labels in val_bar:
            with torch.no_grad():
                logits = model(features)
            predictions = logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, labels))
            accurate_preds = predictions == references
            num_elements += accurate_preds.shape[0]
            accurate += accurate_preds.long().sum()

            if accelerator.is_local_main_process:
                val_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation')

        # 保存模型
        if accelerator.is_main_process:
            # 计算loss和acc并保存到tensorboard
            train_loss = total_loss / (len(train_loader) * config.trainer.batch_size)
            eval_metric = accurate.item() / num_elements

            print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] loss = {train_loss:.5f}, acc = {100 * eval_metric:.5f} %")
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Val Acc', eval_metric, epoch)
            if eval_metric > best_acc:
                best_acc = eval_metric
                print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] 保存模型")
                save_model(model, config.model.save_name)
                stale = 0
            else:
                stale += 1
                if stale > patience:
                    print(f"连续的 {patience}  epochs 模型没有提升，停止训练")
                    accelerator.end_training()
                    break

    print(f"最高acc: {best_acc:.5f}")
    # 模型最后验证
    # 还没写
