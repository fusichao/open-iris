import numpy as np
import matplotlib.pyplot as plt


def get_quadratic_vertex(y_values):
    """
    输入: y_values 为长度为 3 的数组/列表，代表 x 为 [-1, 0, 1] 处的 y 值
    输出: 拟合曲线顶点的 x 坐标
    """
    y_neg1, y_0, y_1 = y_values

    # 计算系数 a 和 b
    # a = (y1 + y-1 - 2*y0) / 2
    # b = (y1 - y-1) / 2

    numerator = y_neg1 - y_1
    denominator = 2 * (y_1 + y_neg1 - 2 * y_0)

    # 检查分母是否为 0（即三点共线，无法构成二次曲线）
    if denominator == 0:
        return float('inf')  # 或者返回 None，取决于你的业务逻辑

    return numerator / denominator



def get_min_first_col_idx_and_second_val(x: np.ndarray):
    """
    x: shape (n, 2) 的 ndarray
    返回:
        idx: 使 x[:, 0] 最小的下标
        val: x[idx, 1] 的值
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("x 必须是形状为 (n, 2) 的 ndarray")

    idx = np.argmin(x[:, 0])
    return idx, x[idx, 1]


if __name__ == "__main__":
    y_input = [4, 1, 1]
    vertex_x = get_quadratic_vertex(y_input)
    print(f"顶点 x 坐标为: {vertex_x}")

    data1 = np.load("./data/video1_360.npy")
    print(data1.shape)
    all_y = []
    for i in range(data1.shape[0]):
        x = data1[i]
        idx = np.argmin(x[:, 0])
        if 0 < idx < x.shape[0] - 1:
            vertex = get_quadratic_vertex(x[idx - 1:idx + 2, 0])
        else:
            vertex = 0
        y = x[idx, 1] + vertex
        all_y.append(y)

    all_y = np.array(all_y)

    plt.figure(figsize=(8, 6))

    # 绘制曲线
    plt.plot(all_y[:181], 'b-', linewidth=2)

    # 设置轴标签
    plt.xlabel('Frame No', fontsize=12)
    plt.ylabel('Rotate Angle(degree)', fontsize=12)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 添加标题
    plt.title('IRIS Rotate', fontsize=14)

    # 自动调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()


    bk = 6