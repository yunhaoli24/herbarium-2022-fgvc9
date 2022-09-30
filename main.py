import torch
import torch.nn as nn
import yaml
from torchvision.models import resnet18
from easydict import EasyDict
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from accelerate import Accelerator
from src.utils import same_seeds, save_model

if __name__ == '__main__':
    # 读取配置
    config = EasyDict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(42)
    accelerator = Accelerator()
    if accelerator.is_main_process():
        print(config)

    # TODO: 加载数据
    train_loader = None
    val_loader = None
    num_classes = None

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    model = resnet18(pretrained=True, num_classes=num_classes)
    if config.trainer.resume:
        model.load_state_dict(torch.load(config.model.checkpoint_path, map_location=torch.device('cpu')))

    # 定义训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.lr)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.trainer.num_epochs / 10), eta_min=1e-5)

    model, optimizer, criterion, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, criterion, scheduler, train_loader, val_loader)

    # Tensorboard
    writer = SummaryWriter()

    stale = 0
    best_acc = 0
    patience = config.trainer.num_epochs / 2
    # 开始训练
    if accelerator.is_main_process():
        print("开始训练！")

    for epoch in range(config.trainer.num_epochs):
        model.train()

        # 训练
        total_loss = 0
        train_bar = tqdm(train_loader)
        for features, labels in train_bar:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            optimizer.step()

            train_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training')

        # 验证
        model.eval()
        total_correct = 0
        val_bar = tqdm(val_loader)
        for features, labels in val_bar:
            with torch.no_grad():
                logits = model(features)
            pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            total_correct += torch.sum(pred == labels).item()

            val_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation')

        # 保存模型
        if accelerator.is_main_process():
            # 计算loss和acc并保存到tensorboard
            train_loss = total_loss / (len(train_loader) * config.trainer.batch_size)
            val_acc = total_correct / (len(val_loader) * config.trainer.batch_size)

            print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] loss = {train_loss:.5f}, acc = {val_acc:.5f}")
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Val Acc', val_acc, epoch)
            scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc
                print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] 保存模型")
                save_model(model, config.model.checkpoint_path)
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
