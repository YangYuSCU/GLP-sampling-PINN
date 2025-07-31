import logging
import sys
from init_config import get_device, path, root_path, init_log, train_Adam,TASK_NAME
from model_config import PINNConfig
from train_config import *
from scipy.interpolate import griddata
from matplotlib import gridspec, pyplot as plt
from plot.heatmap import plot_heatmap
# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)


if __name__ == "__main__":
    # 设置需要写日志
    init_log()

    # cuda 调用
    device = get_device(sys.argv)

    log(layers)

    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    # 打印配置
    config_str = 'Train_Grid_Size ' + str(Train_Grid_Size) 
    log(config_str)

    train_dict = {
        'x_region': x_grid,
        'y_region': y_grid,
        'X_b_down': X_b_down,
        'X_b_up': X_b_up,
        'X_b_left': X_b_left,
        'X_b_right': X_b_right,
        'u_b_down': u_b_down,
        'u_b_up': u_b_up,
        'u_b_left': u_b_left,
        'u_b_right': u_b_right,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'u_valid': u_valid,
        'p': p,
    }

    #画出采样点
    #设置横坐标刻度
    plt.xlim(-1.0, 1.0)
    #设置纵坐标刻度
    plt.ylim(-1.0,1.0)
    plt.gca().set_aspect(1)
    plt.plot(x_grid, y_grid,
            color='red',  # 全部点设置为红色
            marker='.', markersize=0.6, # 点的形状为圆点
            linestyle='', linewidth=1)  # 线型为空，也即点与点之间不用线连接

    file_name0 = root_path + '/' + TASK_NAME+ '/' 
    plt.savefig(file_name0 + 'points.JPEG', dpi=500)
    plt.clf()


    ##训练
    train_Adam(layers, device, param_dict, train_dict, Adam_steps=500)




    # 加载模型
    net_path = root_path + '/' + path + '/PINN.pkl'
    model_config = PINNConfig.reload_config(net_path=net_path)

    # 预测网格点
    PRED_GRID_SIZE = 400
    X_PRED_GRID = np.linspace(lb[0], ub[0], PRED_GRID_SIZE)
    Y_PRED_GRID = np.linspace(lb[1], ub[1], PRED_GRID_SIZE)
    X_PRED, Y_PRED = np.meshgrid(X_PRED_GRID, Y_PRED_GRID)
    X_pred = np.hstack((X_PRED.flatten()[:, None], Y_PRED.flatten()[:, None]))
    u_pred = Exact_u_func(X_pred[:, 0:1], X_pred[:, 1:2])

    X_star = X_pred
    u_star = u_pred
    
    x_star = model_config.data_loader(X_pred[:, 0:1])
    y_star = model_config.data_loader(X_pred[:, 1:2])
    u_pred = model_config.net_u(x_star, y_star)

    u_pred = model_config.detach(u_pred)
    x_star = model_config.detach(x_star)
    y_star = model_config.detach(y_star)

    # 计算误差
    RE_Linfinity = np.linalg.norm(u_star-u_pred,np.inf)/np.linalg.norm(u_star,np.inf)
    RE_L2 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)


    log('FINAL_RE_Linfinity: %e' % (RE_Linfinity))
    log('FINAL_RE_L2: %e' % (RE_L2))


    U_star = griddata(X_star, u_star.flatten(), (X_PRED, Y_PRED), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X_PRED, Y_PRED), method='cubic')
    
    file_name1 = root_path + '/' + TASK_NAME + '/' + '/heatmap_true'
    file_name2 = root_path + '/' + TASK_NAME + '/' + '/heatmap_pred'
    file_name3 = root_path + '/' + TASK_NAME + '/' + '/heatmap_error'
    plot_heatmap(X=X_PRED, Y=Y_PRED, Z=U_star, xlabel='x',
                     ylabel='y', file_name=file_name1)
    plot_heatmap(X=X_PRED, Y=Y_PRED, Z=U_pred, xlabel='x',
                     ylabel='y', file_name=file_name2)
    plot_heatmap(X=X_PRED, Y=Y_PRED, Z=np.abs(U_star-U_pred), xlabel='x',
                     ylabel='y', file_name=file_name3)    
    

