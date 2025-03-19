import numpy as np
import matplotlib.pyplot as plt

class HIVModel:
    def __init__(self, A, alpha, B, beta):

        self.A = A
        self.alpha = alpha
        self.B = B
        self.beta = beta        

    def viral_load(self, time):
        # TODO: 计算病毒载量

        return self.A * np.exp(-self.alpha * time) + self.B * (1 - np.exp(-self.beta * time))

    def plot_model(self, time):
        # TODO: 绘制模型曲线
        viral_loads = self.viral_load(time)
        plt.figure(figsize=(10, 6))
        plt.plot(time, viral_loads, label="病毒载量", color="blue")
        plt.title("HIV病毒载量模型")
        plt.xlabel("时间")
        plt.ylabel("病毒载量")
        plt.legend()
        plt.grid()
        plt.show()

def load_hiv_data(filepath):
    # TODO: 加载HIV数据
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        time = data[:, 0]
        viral_load = data[:, 1]
        return time, viral_load
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return np.array([]), np.array([])

    

def main():
    # TODO: 主函数，用于测试模型
    A = 100
    alpha = 0.1
    B = 1000
    beta = 0.05

    # 创建模型实例
    model = HIVModel(A, alpha, B, beta)

    # 定义时间范围
    time = np.linspace(0, 100, 500)

    # 绘制模型曲线
    model.plot_model(time)

    # 加载实际数据并绘制对比
    filepath = "data/HIVseries.csv"  # 数据文件名
    actual_time, actual_viral_load = load_hiv_data(filepath)
    if actual_time.size > 0 and actual_viral_load.size > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(time, model.viral_load(time), label="模型预测", color="blue")
        plt.scatter(actual_time, actual_viral_load, label="实际数据", color="red")
        plt.title("HIV病毒载量模型与实际数据对比")
        plt.xlabel("时间")
        plt.ylabel("病毒载量")
        plt.legend()
        plt.grid()
        plt.show()   

if __name__ == "__main__":
    main()
