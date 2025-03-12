from typing import List
import tqdm
from mmengine.evaluator import Evaluator
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import _ParamScheduler
from mmengine.model.base_model.base_model import BaseModel
from torch.utils.data.dataloader import DataLoader


class Runner:
    def __init__(self, model: BaseModel, optimizer: OptimWrapper, schedulers: List[_ParamScheduler], evaluator: Evaluator):
        self.model = model
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.evaluator = evaluator

    def run_train(self, loader: DataLoader, epoch: int = 0):
        self.model.train()
        n = len(loader)
        for i, data_samples in enumerate(tqdm.tqdm(loader)):
            self.model.train_step(data_samples, optim_wrapper=self.optimizer)
            # 可以多 batch 共同计算 loss，但与 train_step 冲突，需要手写 train_step 的内部代码，其内部代码结构如下：
            # def train_step(...):
            #     with optim_wrapper.optim_context(self):
            #         data = self.data_preprocessor(data, True)
            #         losses = self._run_forward(data, mode='loss')  # type: ignore
            #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            #     optim_wrapper.update_params(parsed_losses)
            #     return log_vars
            # i += 1
            # if i == n or i % config.batch_loss_count == 0:
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()

    def run_test(self, loader: DataLoader, epoch: int = 0):
        self.model.eval()
        n = len(loader)
        for i, data_samples in enumerate(tqdm.tqdm(loader)):
            outputs = self.model.val_step(data_samples)
            print(len(outputs))
            # 评估怎么写还需要认真考虑一下
        #     self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        # metrics = self.evaluator.evaluate(n)
        # print(metrics)
