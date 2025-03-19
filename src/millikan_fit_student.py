import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    """
    加载 TXT 数据文件
    
    参数:
        filename: 数据文件路径
        
    返回:
        x: 频率数据数组
        y: 电压数据数组
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 {filename} 未找到！请检查文件路径。")

    try:
        # 先尝试用逗号分隔符读取
        data = np.loadtxt(filename, delimiter=',')
    except ValueError:
        try:
            # 若失败，尝试空格或制表符
            data = np.loadtxt(filename)  # 自动检测空格、制表符
        except Exception as e:
            raise ValueError(f"无法解析 {filename}，请检查数据格式！错误信息: {e}")

    if data.shape[1] < 2:
        raise ValueError("数据文件至少应包含两列：频率 和 电压")

    x = data[:, 0]  # 第一列：频率
    y = data[:, 1]  # 第二列：电压
    return x, y

def calculate_parameters(x, y):
    """
    计算最小二乘拟合参数
    
    参数:
        x: x坐标数组
        y: y坐标数组
        
    返回:
        m: 斜率
        c: 截距
        Ex: x的平均值
        Ey: y的平均值
        Exx: x^2的平均值
        Exy: xy的平均值
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("数据点数量不足，至少需要 2 个数据点进行拟合！")

    Ex = np.mean(x)
    Ey = np.mean(y)
    Exx = np.mean(x**2)
    Exy = np.mean(x*y)
    
    denominator = Exx - Ex**2
    if denominator == 0:
        raise ValueError("数据点存在问题，导致分母为零，无法计算拟合参数！")

    m = (Exy - Ex * Ey) / denominator
    c = (Exx * Ey - Ex * Exy) / denominator
    return m, c, Ex, Ey, Exx, Exy

def plot_data_and_fit(x, y, m, c):
    """
    绘制数据点和拟合直线
    
    参数:
        x: x坐标数组
        y: y坐标数组
        m: 斜率
        c: 截距
    
    返回:
        fig: matplotlib 图像对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 画散点图
    ax.scatter(x, y, color='blue', label='experimental data', zorder=3)

    # 生成拟合直线（扩展 x 取值范围）
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = m * x_fit + c
    ax.plot(x_fit, y_fit, color='red', linestyle='--', label='fitting curve', zorder=2)

    # 设置标题和标签
    ax.set_xlabel('frequency(Hz)')
    ax.set_ylabel('voltage(V)')
    ax.set_title('Fitting of experimental data for photoelectric effect')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig

def calculate_planck_constant(m):
    """
    计算普朗克常量
    
    参数:
        m: 斜率
        
    返回:
        h: 计算得到的普朗克常量值
        relative_error: 与实际值的相对误差(%)
    """
    e = 1.602e-19  # 电子电荷 (C)
    h_actual = 6.626e-34  # 普朗克常量 (J·s)

    h = m * e
    relative_error = abs(h - h_actual) / h_actual * 100  # 计算相对误差

    return h, relative_error

def main():
    """主函数"""
    filename = "millikan.txt"  # 数据文件路径
    
    try:
        # 加载数据
        x, y = load_data(filename)
        
        # 计算拟合参数
        m, c, Ex, Ey, Exx, Exy = calculate_parameters(x, y)
        
        # 打印结果
        print(f"Ex = {Ex:.6e}")
        print(f"Ey = {Ey:.6e}")
        print(f"Exx = {Exx:.6e}")
        print(f"Exy = {Exy:.6e}")
        print(f"斜率 m = {m:.6e}")
        print(f"截距 c = {c:.6e}")
        
        # 绘制数据和拟合直线
        fig = plot_data_and_fit(x, y, m, c)
        
        # 计算普朗克常量
        h, relative_error = calculate_planck_constant(m)
        print(f"计算得到的普朗克常量 h = {h:.6e} J·s")
        print(f"与实际值的相对误差: {relative_error:.2f}%")
        
        # 保存图像
        fig.savefig("millikan_fit.png", dpi=300)
        plt.show()

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
