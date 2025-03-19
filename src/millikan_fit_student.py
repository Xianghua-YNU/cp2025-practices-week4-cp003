"""
最小二乘拟合和光电效应实验
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

def load_data(filename):
    """
    加载数据文件
    
    参数:
        filename: 数据文件路径
        
    返回:
        x: 频率数据数组
        y: 电压数据数组
    """
    try:
        data = np.loadtxt(filename)
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except ValueError as e:
        print(f"Error loading data: {e}")
        return None, None

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
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays cannot be empty")
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    Ex = np.mean(x)
    Ey = np.mean(y)
    Exx = np.mean(x**2)
    Exy = np.mean(x * y)
    if Exx - Ex**2 == 0:
        raise ValueError("Variance of x is zero, cannot calculate slope")
    m = (Exy - Ex * Ey) / (Exx - Ex**2)
    c = (Exx * Ey - Ex * Exy) / (Exx - Ex**2)
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
        fig: matplotlib图像对象
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    if len(x) == 0:
        raise ValueError("x and y arrays cannot be empty")
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(x, m * x + c, color='red', label="Fit")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
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
    if m == 0:
        raise ValueError("Slope cannot be zero")
    e = 1.602e-19  # 电子电荷
    h = e * m
    h_actual = 6.626e-34  # J·s
    relative_error = abs(h - h_actual) / h_actual * 100
    return h, relative_error

def main():
    """主函数"""
    # 数据文件路径
    filename = "/workspaces/cp2025-practices-week4-cp003/data/millikan.txt"
    
    # 加载数据
    x, y = load_data(filename)
    if x is None or y is None:
        return
    
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

if __name__ == "__main__":
    main()

# 测试用例
def test_calculate_planck_constant_invalid_slope():
    """测试无效斜率时的异常处理"""
    with pytest.raises(ValueError):
        calculate_planck_constant(0)  # 斜率为0

def test_plot_data_and_fit_invalid_input():
    """测试无效输入时的异常处理"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2])  # 长度不匹配
    with pytest.raises(ValueError):
        plot_data_and_fit(x, y, 1, 0)
    with pytest.raises(ValueError):
        plot_data_and_fit(np.array([]), np.array([]), 1, 0)  # 空数组
