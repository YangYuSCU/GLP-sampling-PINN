import numpy as np
import torch
import os
import os.path
import scipy.io

def getCurDirName():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # curDirName = curDir.split(parentDir)[-1]
    curDirName = os.path.split(curDir)[-1]
    return curDirName

def getParentDir():
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # this will return parent directory.
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    # parentDirName = os.path.split(parentDir)[-1]
    return parentDir

Rdatapath = getParentDir()+ '/'+getCurDirName()

TASK_NAME = 'Poisson_OnePeak'

#方程参数
p = 1000**(1/2)


# 区域内部网格大小
Train_Grid_Size = 10946



def Exact_u_func(x,y):
    u = np.exp(-p**2*(x**2+y**2))
    return u

# 设置定义域
lb = np.array([-1,-1])
ub = np.array([1,1])

layers = [2, 40, 40, 40, 40,  1]

# 生成训练数据
#读取NTM点
#读取.RData文件
Rdata = np.loadtxt(Rdatapath+'/%dp_2D.txt'%Train_Grid_Size)
x_grid = Rdata[:, 0:1]
y_grid = Rdata[:, 1:2]

# 生成的均匀随机分布的x和y值 
# np.random.seed(100) 
# x_grid = np.random.uniform(low=lb[0], high=ub[0], size=Train_Grid_Size).flatten()[:, None]
# y_grid = np.random.uniform(low=lb[1], high=ub[1], size=Train_Grid_Size).flatten()[:, None]

#读取LHS点
#np.random.seed(100) 
# XY_data = scipy.io.loadmat(Rdatapath + '/XY_LHS.mat')
# x_grid = XY_data['x_grid']
# y_grid = XY_data['y_grid']

# #读取Halton点
# XY_data = scipy.io.loadmat(Rdatapath + '/XY_Halton.mat')
# x_grid = XY_data['x_grid']
# y_grid = XY_data['y_grid']

# #读取Hammersley点
# XY_data = scipy.io.loadmat(Rdatapath + '/XY_Hammersley.mat')
# x_grid = XY_data['x_grid']
# y_grid = XY_data['y_grid']

# #读取Sobol点
# XY_data = scipy.io.loadmat(Rdatapath + '/XY_Sobol.mat')
# x_grid = XY_data['x_grid']
# y_grid = XY_data['y_grid']




#########################################################################################################
Bou_Grid_Size = 250
Xb = np.linspace(lb[0], ub[0], Bou_Grid_Size+2)
Yb = np.linspace(lb[1], ub[1], Bou_Grid_Size+2)

# 边界点传入
# 外边界网格点
X_Boundary1 = []
X_Boundary2 = []
X_Boundary3 = []
X_Boundary4 = []
for i in Xb:
    for j in Yb:
        if i == lb[0]:
            X_Boundary1.append([i, j]) #左边界
        elif i==ub[0]:
            X_Boundary2.append([i, j]) #右边界
        elif j==lb[1]:
            X_Boundary3.append([i, j]) #下边界
        elif j==ub[1]: 
            X_Boundary4.append([i, j]) #上边界
X_Boundary1 = np.asarray(X_Boundary1, dtype=float)
X_Boundary2 = np.asarray(X_Boundary2, dtype=float)
X_Boundary3 = np.asarray(X_Boundary3, dtype=float)
X_Boundary4 = np.asarray(X_Boundary4, dtype=float)


#采样全部边界点
X_b_left  = X_Boundary1
X_b_right = X_Boundary2
X_b_down = X_Boundary3
X_b_up = X_Boundary4

u_b_left = Exact_u_func(X_b_left[:, 0:1],X_b_left[:, 1:2])
u_b_right = Exact_u_func(X_b_right[:, 0:1],X_b_right[:, 1:2])
u_b_down = Exact_u_func(X_b_down[:, 0:1],X_b_down[:, 1:2])
u_b_up = Exact_u_func(X_b_up[:, 0:1],X_b_up[:, 1:2])  



PRED_GRID_SIZE = 400
X_PRED_GRID = np.linspace(lb[0], ub[0], PRED_GRID_SIZE)
Y_PRED_GRID = np.linspace(lb[1], ub[1], PRED_GRID_SIZE)
X_PRED, Y_PRED = np.meshgrid(X_PRED_GRID, Y_PRED_GRID)
x_valid = X_PRED.flatten()[:, None]
y_valid = Y_PRED.flatten()[:, None]
u_valid = Exact_u_func(x_valid,y_valid)