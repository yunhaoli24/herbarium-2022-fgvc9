import os
from datetime import datetime
from functools import partial

import evaluate
import pytz
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader
from src.models.models_vit import VisionTransformer
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger


def train_one_epoch(model: torch.nn.Module, loss_function: torch.nn.modules.loss._Loss, train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    accelerator: Accelerator, epoch: int, step: int):
    # 训练
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):

        # 第5轮之后加入全部参数
        if epoch > 5:
            for key, params in model.named_parameters():
                params.requires_grad = True

        image, label_dict = batch[0], batch[1]
        category, genus, institution = model(image)

        loss_category = loss_function(category, label_dict['category_id'])
        loss_genus = loss_function(genus, label_dict['genus_id'])
        loss_institution = loss_function(institution, label_dict['institution_id'])
        total_loss = loss_category + loss_genus + loss_institution
        accelerator.log({
            'loss_category': float(loss_category),
            'loss_genus': float(loss_genus),
            'loss_institution': float(loss_institution),
            'total_loss': float(total_loss),
        }, step=step)

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train Total Loss': float(total_loss),
        }, step=step)
        step += 1

        accelerator.print(
            f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i}/{len(train_loader)}] Total Loss: {total_loss:1.5f} loss_category {float(loss_category)} loss_genus {float(loss_genus)} loss_institution {float(loss_institution)}'
        )

    scheduler.step(epoch)
    return total_loss, step


def val_one_epoch(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader,
                  config: EasyDict, accelerator: Accelerator):
    # 验证
    model.eval()
    evaluator = evaluate.load("accuracy")
    for i, batch in enumerate(val_loader):
        features, label_dict = batch[0], batch[1]
        with torch.no_grad():
            category, genus, institution = model(features)
        predictions = category.argmax(dim=-1)
        evaluator.add_batch(predictions=predictions, references=label_dict['category_id'])
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

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    accelerator.print('加载模型...')
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_classes=config.model.num_classes,
        drop_path_rate=0.3, global_pool=True, genus=config.model.genus, institution=config.model.institution)
    # 加载预训练模型
    model.load_state_dict(torch.load('checkpoint-18.pth')['model'], strict=False)
    # 设置只有头可以训练
    for key, params in model.named_parameters():
        if not key.endswith('head'):
            params.requires_grad = False

    accelerator.print('加载数据集...')
    train_loader, val_loader = get_dataloader(config)

    # 定义训练参数
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup, max_epochs=config.trainer.num_epochs)
    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0
    step = 0
    starting_epoch = 0

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # 尝试继续训练
    if config.trainer.resume:
        starting_epoch, step = utils.resume_train_state(config.trainer.save_dir, train_loader, accelerator)

    # 开始训练
    accelerator.print("开始训练！")

    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # 训练
        train_loss, step = train_one_epoch(model, loss_function, train_loader, optimizer, scheduler, accelerator, epoch, step)
        # 验证
        acc = val_one_epoch(model, val_loader, config, accelerator)

        accelerator.print(f'mean acc: {100 * acc:.5f}%')
        accelerator.print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()}, loss = {train_loss:.5f}, acc = {100 * acc:.5f} %")

        # 保存模型
        if acc > best_acc:
            accelerator.save_state(output_dir=f"{os.getcwd()}/{config.trainer.save_dir}/best")
            best_acc = acc

        accelerator.save_state(output_dir=f"{os.getcwd()}/{config.trainer.save_dir}/epoch_{epoch}")

    accelerator.print(f"最高acc: {best_acc:.5f}")