import torch.nn as nn        

class MLP(nn.Module):
    def __init__(self, layers, act_func=nn.Tanh()):
        super().__init__()
        self.layers = layers
        self.act_func = act_func
        self.linear_list = []
        for i in range(len(self.layers)-2):
            linear = nn.Linear(self.layers[i],self.layers[i+1])
            self.weight_init(linear)
            self.linear_list.append(linear)
        self.linear_list = nn.ModuleList(self.linear_list)
        linear = nn.Linear(self.layers[-2],self.layers[-1])
        self.weight_init(linear)
        self.fc = linear

    def forward(self, x):
        for i in range(len(self.linear_list)):
            linear = self.linear_list[i]
            x = self.act_func(linear(x))
        linear = self.fc
        y = linear(x)
        return y

    # 模型初始化
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式 
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

