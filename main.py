import evaluate
import timm
import torch
import yaml
import torch.nn.functional as F
from easydict import EasyDict
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from accelerate import Accelerator

from src.loader import get_dataloader
from src.utils import same_seeds, save_model, load_model

if __name__ == '__main__':
    # 读取配置
    accelerator = Accelerator()
    config = EasyDict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(42)
    accelerator.print(config)

    accelerator.print('加载数据集...')
    train_loader, val_loader = get_dataloader(config)

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    accelerator.print('加载模型')
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=config.model.num_classes)

    # 定义训练参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.lr)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.trainer.num_epochs / 10), eta_min=1e-5)

    # Tensorboard
    writer = SummaryWriter() if accelerator.is_local_main_process else None

    start_epoch = 0
    stale = 0
    best_acc = 0
    step = 0
    patience = config.trainer.num_epochs / 2

    # 尝试继续训练
    if config.trainer.resume:
        accelerator.print(f'从 {config.model.save_name} 加载预训练模型')
        model, optimizer, scheduler, start_epoch = load_model(config.model.save_name, model, optimizer, scheduler)
        accelerator.print(f'加载训练状态成功！从 epoch {start_epoch + 1} 开始训练')

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # 开始训练
    accelerator.print("开始训练！")

    for epoch in range(start_epoch, config.trainer.num_epochs):

        # 训练
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, disable=not accelerator.is_local_main_process)
        if accelerator.is_local_main_process:
            train_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training')
        for features, labels in train_bar:
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if accelerator.is_local_main_process:
                train_loss = '{0:1.5f}'.format(loss)
                train_bar.set_postfix({'loss': f'{train_loss}'})
                writer.add_scalar('Train Loss', loss, step)
                step += 1

        # 验证
        model.eval()
        evaluator = evaluate.load("accuracy")
        val_bar = tqdm(val_loader, disable=not accelerator.is_local_main_process)
        if accelerator.is_local_main_process:
            val_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation')
        for features, labels in val_bar:
            with torch.no_grad():
                logits = model(features)
            predictions = logits.argmax(dim=-1)
            evaluator.add_batch(predictions=predictions, references=labels)

        # 计算loss和acc并保存到tensorboard
        evaluate_result = evaluator.compute()
        accelerator.print(evaluate_result)

        accelerator.print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] loss = {train_loss}, acc = {100 * evaluate_result['accuracy']:.5f} %")

        if accelerator.is_local_main_process:
            writer.add_scalar('Val Acc', evaluate_result['accuracy'], epoch)

        # 保存模型
        if evaluate_result['accuracy'] > best_acc:
            best_acc = evaluate_result['accuracy']
            accelerator.print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] 保存模型")
            # 保存模型
            save_model(config.model.save_name, epoch, model, optimizer, scheduler, accelerator)
            stale = 0
        else:
            stale += 1
            if stale > patience:
                accelerator.print(f"连续的 {patience}  epochs 模型没有提升，停止训练")
                accelerator.end_training()
                break

    accelerator.print(f"最高acc: {best_acc:.5f}")
