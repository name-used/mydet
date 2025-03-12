from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from mmengine.dataset.utils import pseudo_collate
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmdet.registry import MODELS, PARAM_SCHEDULERS, EVALUATOR

import config
import configs
from .runner import Runner
from .dataset import Dataset


def train(work_root: Path):
    # 数据集
    train_dataset = Dataset(pipeline=configs.train_pipeline, **configs.train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=pseudo_collate)
    valid_dataset = Dataset(pipeline=configs.valid_pipeline, **configs.valid)
    valid_dataset.sample_by_step()
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=pseudo_collate)
    # 加载模型
    model = MODELS.build(configs.model)
    if config.load_from:
        state_dict = torch.load(config.load_from)['state_dict']
        # 与 num_class 相关的参数需要被重新训练，不需要加载，且加载会报错
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith('bbox_head') or key == 'dn_query_generator.label_embedding.weight':
                state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
    model.cuda()

    # 加载训练器
    optimizer = build_optim_wrapper(model, configs.optimizer)
    optimizer.initialize_count_status(model, 0, len(train_loader))  # type: ignore
    schedulers = [
        PARAM_SCHEDULERS.build(sch, default_args=dict(optimizer=optimizer, epoch_length=len(train_loader)))
        for sch in configs.schedulers
    ]

    # 加载评估器
    # evaluator = EVALUATOR.build(config.evaluator)

    runner = Runner(model, optimizer, schedulers, evaluator=None)

    # 训练项
    for epoch in range(config.epoch):
        epoch += 1
        print(rf'Train Epoch: {epoch}')
        # 每次训练集重采样
        train_dataset.sample_random()
        runner.run_train(train_loader, epoch)
        if epoch % config.interval == 0:
            runner.run_test(valid_loader, epoch)
        torch.save(model.state_dict(), work_root / rf'epoch_{epoch}.ckpt')
