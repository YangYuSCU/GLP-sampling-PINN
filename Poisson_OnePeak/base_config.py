import os
import torch
import numpy as np
from torch import autograd



torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)

class BaseConfig:
    def __init__(self):
        super().__init__()
        # 训练用的参数
        self.loss = None
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

    # 数据加载函数，转变函数类型及其使用设备设置
    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float64)
        return x_tensor.to(self.device)

    # 左边归一化函数，[-1, 1], lb:下确界，ub:上确界
    def coor_shift(self, X, lb, ub):
        X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
        # X_shift = torch.from_numpy(X_shift).float().requires_grad_()
        return X_shift

    # 将数据从设备上取出
    def detach(self, data):
        tmp_data = data.detach().cpu().numpy()
        if np.isnan(tmp_data).any():
            raise Exception
        return tmp_data

    # 损失函数计算损失并返回
    def loss_func(self, pred_, true_=None):
        # 采用MSELoss
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
            # true_ = self.data_loader(true_)
        return self.loss_fn(pred_, true_)

    # 直接计算一阶导数
    def compute_grad(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x
        
    # 训练一次
    def optimize_one_epoch(self):
        return self.loss

    def train_Adam(self, params, Adam_steps = 50000, Adam_init_lr = 1e-3, scheduler_name=None, scheduler_params=None):
        Adam_optimizer = torch.optim.Adam(params=params,
                                        lr=Adam_init_lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                      amsgrad=False)
        self.optimizer = Adam_optimizer
        self.optimizer_name = 'Adam'
        if scheduler_name == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            Adam_scheduler = MultiStepLR(Adam_optimizer, **scheduler_params)
        else:
            Adam_scheduler = None
        self.scheduler = Adam_scheduler
        for it in range(Adam_steps):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()


    # 保存模型
    # @staticmethod
    def save(net, path, name='PINN'):
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存神经网络
        torch.save(net, path + '/' + name + '.pkl')  # 保存整个神经网络的结构和模型参数


    # 载入整个神经网络的结构及其模型参数
    @staticmethod
    def reload_config(net_path):
        # 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构
        net = torch.load(net_path,weights_only=False)
        return net

