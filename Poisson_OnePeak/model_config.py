import logging
import time
import numpy as np
import torch
from base_config import BaseConfig
from torch.autograd import Variable



# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)


class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model):
        super().__init__()
        self.init()
        self.model = model
        # 设置使用设备:cpu, cuda
        lb, ub, self.device, self.path, self.root_path = self.unzip_param_dict(
            param_dict=param_dict)

        x_region , y_region ,X_b_down, X_b_up, X_b_left, X_b_right,\
        u_b_down, u_b_up, u_b_left, u_b_right,x_valid , y_valid , \
        self.u_valid, p= self.unzip_train_dict(
            train_dict=train_dict)

        # 特殊参数
        self.p = p

        # 区域内点
        self.x_region = self.data_loader(x_region)
        self.y_region = self.data_loader(y_region)
        
        # 边界点
        self.x_b_down  = self.data_loader(X_b_down[:, 0:1])
        self.y_b_down  = self.data_loader(X_b_down[:, 1:2])
        
        self.x_b_up  = self.data_loader(X_b_up[:, 0:1])
        self.y_b_up  = self.data_loader(X_b_up[:, 1:2])
        
        self.x_b_left  = self.data_loader(X_b_left[:, 0:1])
        self.y_b_left  = self.data_loader(X_b_left[:, 1:2])
        
        self.x_b_right  = self.data_loader(X_b_right[:, 0:1])
        self.y_b_right  = self.data_loader(X_b_right[:, 1:2])

        # 验证点
        self.x_valid = self.data_loader(x_valid)
        self.y_valid = self.data_loader(y_valid)

        #边界条件
        self.u_b_down  = self.data_loader(u_b_down)
        self.u_b_up  = self.data_loader(u_b_up)
        self.u_b_left = self.data_loader(u_b_left)
        self.u_b_right  = self.data_loader(u_b_right)

        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)


        self.params = list(self.model.parameters())



    # 训练参数初始化
    def init(self, loss_name='mean', model_name='PINN'):
        self.start_time = None
        # 小于这个数是开始保存模型
        self.min_loss = 1e20
        # 记录运行步数
        self.nIter = 0
        # 损失计算方式
        if loss_name == 'sum':
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        else:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        # 保存模型的名字
        self.model_name = model_name


    # 参数读取
    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['lb'], param_dict['ub'],
                      param_dict['device'], param_dict['path'],
                      param_dict['root_path'])
        return param_data

    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x_region'],
            train_dict['y_region'],
            train_dict['X_b_down'],
            train_dict['X_b_up'],
            train_dict['X_b_left'],
            train_dict['X_b_right'],
            train_dict['u_b_down'],
            train_dict['u_b_up'],
            train_dict['u_b_left'],
            train_dict['u_b_right'],          
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['u_valid'],
            train_dict['p']
        )
        return train_data

    
    def model_params(self):
        return list(self.model.parameters())
    
    def net_u(self, x , y):
        X = torch.cat((x, y), 1)
        X = self.coor_shift(X, self.lb, self.ub)
        u = self.model.forward(X)
        return u
    

    
    def s(self,x,y,p):
        if p == 1000**(1/2):    
            s = 4000000*x**2*torch.exp(-1000*x**2 - 1000*y**2) + 4000000*y**2*torch.exp(-1000*x**2 - 1000*y**2) - 4000*torch.exp(-1000*x**2 - 1000*y**2)
        return -s
    

    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()

        # 训练点
        x_region = self.x_region
        y_region = self.y_region
        u_region = self.net_u(x_region,y_region)
        u_region_x = self.compute_grad(u_region, x_region)
        u_region_y = self.compute_grad(u_region, y_region)
        u_region_yy = self.compute_grad(u_region_y, y_region)
        u_region_xx = self.compute_grad(u_region_x, x_region)
        f_u = u_region_xx+ u_region_yy + self.s(x_region,y_region,self.p)
        
        # 方程损失
        loss_res = self.loss_func(f_u)

        # 边界点
        use_boundary = True
        loss_boundary = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        loss_boundary.requires_grad_()
        if use_boundary:
            x_b_down = self.x_b_down
            y_b_down = self.y_b_down
            u_b_down  = self.net_u(x_b_down,y_b_down)

              
            x_b_up = self.x_b_up
            y_b_up = self.y_b_up
            u_b_up  = self.net_u(x_b_up,y_b_up)    
            
            x_b_left = self.x_b_left
            y_b_left = self.y_b_left
            u_b_left  = self.net_u(x_b_left,y_b_left)

              
            x_b_right = self.x_b_right
            y_b_right = self.y_b_right
            u_b_right  = self.net_u(x_b_right,y_b_right)
            loss_boundary = self.loss_func(u_b_up, self.u_b_up)+ self.loss_func(u_b_down, self.u_b_down) + \
                            self.loss_func(u_b_left, self.u_b_left)+ self.loss_func(u_b_right, self.u_b_right)
            

        # 权重
        alpha_res = 1
        alpha_boundary = 1
        
        self.loss = loss_res * alpha_res + alpha_boundary *loss_boundary
        
        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1

        # 保存模型
        loss = self.detach(self.loss)
        if loss < self.min_loss:
            self.min_loss = loss
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name)

        # 打印日志
        loss_remainder = 100
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_res = self.detach(loss_res)
            loss_boundary = self.detach(loss_boundary)

            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' loss_res ' + str(loss_res) + ' loss_boundary ' + str(loss_boundary) +\
                ' LR ' + str(self.optimizer.state_dict()['param_groups'][0]['lr']) +\
                ' min_loss ' + str(self.min_loss)
            log(log_str)

            # 验证
            with torch.no_grad():
                u_valid = self.net_u(self.x_valid,self.y_valid)
                u_valid = self.detach(u_valid)
                L_infinity = np.linalg.norm(self.u_valid - u_valid, np.inf)
                L_2 = np.linalg.norm(self.u_valid - u_valid, ord=2)
                RE_L2 = L_2/np.linalg.norm(self.u_valid, ord=2)
                RE_Linfinity = L_infinity / \
                    np.linalg.norm(self.u_valid, np.inf)

                error_str = 'L_infinity ' + str(L_infinity) + ' L_2 ' + str(
                    L_2) + ' RE_L2 ' + str(RE_L2) + ' RE_Linfinity ' + str(RE_Linfinity)
                log(error_str)

            # 打印耗时
            elapsed = time.time() - self.start_time
            log('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            self.start_time = time.time()
        return self.loss
