from ActivationFunction import sigmoid,tanh,relu,leaky_relu,elu,swish
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_activation_function(func_dict,x_range=(-4,4),figsize=(12,8)):
    """
    :param func_dic:
    :param x_range:
    :param figsize:
    """
    matplotlib.use('TkAgg')  # 或者尝试 'TkAgg'、'Qt5Agg' 等
    x = np.linspace(x_range[0],x_range[1],500)

    # 自动计算子图布局
    num_plots = len(func_dict)
    rows = int(np.sqrt(num_plots))
    cols = int(np.ceil(num_plots / rows))

    # 创建全局大图（包含所有子图）
    global_fig = plt.figure(figsize=figsize)

    # 4K分辨率的设置 (3840x2160)
    dpi = 300
    single_figsize = (3840 / dpi, 2160 / dpi)

    # 动态生成图
    for idx, (name, (func, display_name, color)) in enumerate(func_dict.items(), 1):
        # 在大图中添加子图
        plt.figure(global_fig.number)
        plt.subplot(rows, cols, idx)
        plt.plot(x, func(x), label=display_name, c=color, lw=2.5)
        plt.title(f"{display_name} Activation")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(visible=True, alpha=0.4)
        plt.legend()

        # 为每个激活函数单独创建一个4K高清图
        single_fig = plt.figure(figsize=single_figsize, dpi=dpi)
        plt.plot(x, func(x), label=display_name, c=color, lw=2.5)
        plt.title(f"{display_name} Activation", fontsize=16)
        plt.xlabel("Input", fontsize=14)
        plt.ylabel("Output", fontsize=14)
        plt.grid(visible=True, alpha=0.4)
        plt.legend(fontsize=12)
        plt.tight_layout()
        # 保存单独的高清图像
        plt.savefig(f"./png/4k_{display_name}.png", dpi=dpi, bbox_inches='tight')
        plt.close(single_fig)  # 关闭单独的图，避免内存问题

    # 返回到全局大图
    plt.figure(global_fig.number)
    plt.tight_layout()
    # 保存全局大图
    plt.savefig(f"./png/all_activation_functions.png", dpi=dpi, bbox_inches='tight')

    # 展示全局大图
    plt.show()


if __name__ == '__main__':
    activation_funcs = {
        "sigmoid": (sigmoid, "Sigmoid", "#FF4E50"),
        "tanh":(tanh, "Tanh", "#FF4E50"),
        "relu": (relu, "Relu", "#FF4E50"),
        "leaky_relu": (leaky_relu, "Leaky ReLU", "#FF4E50"),
        "elu": (elu, "Elu", "#FF4E50"),
        "swish": (swish, "Swish", "#FF4E50"),

    }
    plot_activation_function(activation_funcs)