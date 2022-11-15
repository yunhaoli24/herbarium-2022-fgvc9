import os
from datetime import datetime
from typing import Dict

import evaluate
import pytz
import timm
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from timm.optim import optim_factory
from torch.utils.tensorboard import SummaryWriter

from src import utils
from src.loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger


def train_one_epoch(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss], train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    total_loss = 0

    for i, image, label in enumerate(train_loader):
        seg_result = model(image)

        total_loss = 0
        for name in loss_functions:
            loss = loss_functions[name](seg_result, label)
            accelerator.log({name: float(loss)}, step=step)
            total_loss += loss

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train Total Loss': float(total_loss),
        }, step=step)
        accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i}/{len(train_loader)}] Loss: {total_loss:1.5f}')
        step += 1
    scheduler.step(epoch)
    return total_loss, step


def val_one_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, accelerator: Accelerator):
    # 验证
    model.eval()
    evaluator = evaluate.load("accuracy")
    for i, features, labels in enumerate(val_loader):
        with torch.no_grad():
            logits = model(features)
        predictions = logits.argmax(dim=-1)
        evaluator.add_batch(predictions=predictions, references=labels)
        accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i}/{len(val_loader)}]')

    evaluate_result = evaluator.compute()

    accelerator.log({
        'acc': float(evaluate_result['accuracy']),
    }, step=step)
    return float(evaluate_result['accuracy'])


if __name__ == '__main__':
    # 读取配置
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(42)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d-%H:%M:%S"))
    accelerator = Accelerator(log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('加载数据集...')
    train_loader, val_loader = get_dataloader(config)

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    accelerator.print('加载模型...')
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=config.model.num_classes)

    # 定义训练参数
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup, max_epochs=config.trainer.num_epochs)
    loss_functions = {
        'cross_entropy_loss': torch.nn.CrossEntropyLoss()
    }

    best_acc = 0
    step = 0
    starting_epoch = 0

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # 尝试继续训练
    if config.trainer.resume:
        starting_epoch, step = utils.resume_train_state(config.save_dir, train_loader, accelerator)

    # 开始训练
    accelerator.print("开始训练！")

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # 训练
        train_loss, step = train_one_epoch(model, loss_functions, train_loader, optimizer, scheduler, accelerator, epoch, step)
        # 验证
        acc = val_one_epoch(model, val_loader, config, accelerator)

        accelerator.print(f'mean acc: {100 * acc:.5f}%')
        accelerator.print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()}, loss = {train_loss:.5f}, acc = {100 * acc:.5f} %")

        # 保存模型
        if acc > best_acc:
            accelerator.save_state(output_dir=f"{os.getcwd()}/{config.save_dir}/best")
            best_acc = acc

        accelerator.save_state(output_dir=f"{os.getcwd()}/{config.save_dir}/epoch_{epoch}")

    accelerator.print(f"最高acc: {best_acc:.5f}")