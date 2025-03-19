# 计算物理实验项目：Logistic模型与混沌现象研究

## 项目简介
本项目旨在通过研究著名的Logistic模型，探索混沌现象的基本特性及其在简单数学迭代中的表现。Logistic模型是一个描述自然界生物种群变化的数学模型，其迭代方程为：

$$
x' = rx(1-x)
$$

通过本实验，学生将学习如何实现Logistic模型的迭代，分析不同参数对系统行为的影响，并绘制费根鲍姆图（Feigenbaum Plot），观察从有序到混沌的转变过程。

---

## 实验目标
1. 实现Logistic模型的迭代算法，观察不同参数 $r$ 对系统行为的影响。
2. 分析 $r$ 取不同值时系统的固定点、周期分岔和混沌现象。
3. 绘制费根鲍姆图，研究 $r$ 值对分岔现象的影响，并确定混沌边界。
4. 理解混沌现象的基本特性及其在复杂物理系统中的表现。

---

## 实验任务

### 任务1：Logistic模型的迭代
1. **实验内容**：
   
   • 对Logistic模型进行迭代，初始值设为 $x_0 = 0.5$。
   
   • 分别取 $r=2, 3.2, 3.45, 3.6$，观察前60次迭代中 $x$的变化。
   
   • 在一幅图中绘制四幅子图，分别对应 $r=2, 3.2, 3.45, 3.6$的前60次迭代结果。横坐标为迭代次数，纵坐标为 $x$。

3. **实验要求**：
   
   • 分析不同 $r$值下系统的行为：
   
     ◦ 当 $r=2$时， $x$趋于0.5，结论为**没有分岔**。
   
     ◦ 当 $r=3.2$时， $x$趋于两个值（如0.51304和0.79946），结论为**周期2分岔**。
   
     ◦ 当 $r=3.45$时， $x$趋于四个值，结论为**周期4分岔**。
   
     ◦ 当 $r=3.6$时， $x$的取值没有明确趋向，结论为**混沌**。
import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, n_iter):
    x = np.zeros(n_iter)
    x[0] = x0
    for i in range(1, n_iter):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

# 参数设置
r_values = [2, 3.2, 3.45, 3.6]
x0 = 0.5
n_iter = 60

# 绘制子图
plt.figure(figsize=(12, 8))
for i, r in enumerate(r_values):
    x = logistic_map(r, x0, n_iter)
    plt.subplot(2, 2, i+1)
    plt.plot(range(n_iter), x, 'b-')
    plt.title(f'r = {r}')
    plt.xlabel('Iteration')
    plt.ylabel('x')
plt.tight_layout()
plt.show()
---

### 任务2：费根鲍姆图的绘制
1. **实验内容**：
   
   • 研究 $r$值对分岔现象的影响。
   
   • 设置 $r$的范围为 $[2.6, 4]$，步长为 $0.001$，初始值 $x_0=0.5$。
   
   • 对每个 $r$值进行250次迭代，前100次用于稳定系统，后150次记录 $x$的值。
   
   • 在一幅图中绘制 $(r, x)$的关系图，横坐标为 $r$，纵坐标为 $x$，使用散点图（`k.`或`scatter`）表示。

3. **实验要求**：
   
   • 绘制的图像应呈现倒下的树状结构，即费根鲍姆图。
   
   • 分析费根鲍姆图中固定点、有限循环和混沌的表现形式。
   
   • 确定系统从有序（固定值或有限循环）到混沌的转变点，称为**混沌边界**。

def feigenbaum_plot(r_min, r_max, r_step, x0, n_iter, n_record):
    r_values = np.arange(r_min, r_max, r_step)
    x_values = []
    for r in r_values:
        x = logistic_map(r, x0, n_iter)
        x_values.append(x[n_iter - n_record:])
    return r_values, x_values

# 参数设置
r_min = 2.6
r_max = 4.0
r_step = 0.001
x0 = 0.5
n_iter = 250
n_record = 150

# 生成费根鲍姆图数据
r_values, x_values = feigenbaum_plot(r_min, r_max, r_step, x0, n_iter, n_record)

# 绘制费根鲍姆图
plt.figure(figsize=(10, 6))
for r, x in zip(r_values, x_values):
    plt.plot([r] * len(x), x, 'b,', markersize=1)
plt.title('Feigenbaum Plot')
plt.xlabel('r')
plt.ylabel('x')
plt.show()

## 实验建议
1. **代码优化**：
   
   • 使用Python的数组运算功能（如`numpy`）一次性对所有 $r$值进行迭代计算，以提高计算效率。
   
2. **图像分析**：
   
   • 观察费根鲍姆图中固定点、周期分岔和混沌的表现形式。
   
   • 确定混沌边界的位置，并分析其物理意义。

3. **扩展思考**：
   
   • 混沌现象在自然界中广泛存在，例如流体动力学中的湍流和气象学中的蝴蝶效应。查阅相关资料，了解混沌现象在其他领域的应用。

---

## 实验问题
1. 对于给定的 $r$，费根鲍姆图中的固定点、有限循环和混沌分别是什么样的？
2. 从费根鲍姆图中可以看出，系统从有序到混沌的转变点（混沌边界）大约在 $r$的哪个值？
3. 混沌现象的“确定性”和“随机性”是如何共存的？结合实验结果进行说明。

---

## 提交要求
1. 修改并提交代码文件 `src/logistic_map.py`
2. results/logistic_map_results.md中给出实验结果、图像和问题的回答。

---

## 实验拓展
1. 尝试修改初始值 $x_0$，观察其对迭代结果的影响。
2. 研究其他混沌模型（如Henon映射、Lorenz系统），并与Logistic模型进行比较。
3. 探讨混沌现象在实际物理系统中的应用，例如天气预报、金融市场等。

---

通过本实验，学生将掌握混沌现象的基本理论和研究方法，理解简单数学模型如何揭示复杂系统的行为，并培养编程和数据分析能力。
