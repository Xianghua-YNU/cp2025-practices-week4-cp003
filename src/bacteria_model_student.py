import numpy as np
import matplotlib.pyplot as plt
def W(t, A, tau):
    return A * (np.exp(-t/tau) - A + (A * t))/tau
t = np.linspace(0, 2, 100)
W_values = W(t, A=1, tau=1)
plt.plot(t, W_values, label='W(t), A=1, τ=1')
plt.xlabel('t')
plt.ylabel('W(t)')
plt.title('W(t) with Different Parameters')
plt.xlabel('Time (t)')
plt.ylabel('Function Value')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.grid(True)
plt.show()

params = [
    {'A': 1, 'tau': 1, 'label': 'A=1, τ=1'},
    {'A': 2, 'tau': 0.5, 'label': 'A=2, τ=0.5'},
    {'A': 0.5, 'tau': 2, 'label': 'A=0.5, τ=2'}
]

t = np.linspace(0, 2, 100)
plt.figure()
styles = ['-', '--', '-.', ':']
colors = ['blue', 'green', 'red']
for i, param in enumerate(params):
    W_values = W(t, param['A'], param['tau'])
    plt.plot(t, W_values, linestyle=styles[i], color=colors[i], label=param['label'])
plt.xlabel('t')
plt.ylabel('W(t)')
plt.legend()
plt.grid(True)
plt.show()

# 加载数据

data_a = np.loadtxt('g149novickA.txt')
t_data = data_a[:, 0]
v_data = data_a[:, 1]

# 绘制实验数据点
plt.scatter(t_data, v_data, marker='o', color='red', label='Experimental Data')

# 尝试不同τ值
taus = [1, 2, 3]
t_model = np.linspace(0, max(t_data), 100)

for tau in taus:
    v_model = 1 - np.exp(-t_model/tau)
    plt.plot(t_model, v_model, label=f'τ={tau}')

plt.xlabel('Time (hours)')
plt.ylabel('V(t)')
plt.legend()
plt.show()

# 加载并过滤数据
data_b = np.loadtxt('g149novickB.csv', delimiter=',')
mask = data_b[:, 0] <= 10
t_filtered = data_b[:, 0][mask]
w_filtered = data_b[:, 1][mask]

# 获取大t值的线性部分
large_t = t_filtered[t_filtered > 8]
large_w = w_filtered[t_filtered > 8]

# 线性拟合
coeffs = np.polyfit(large_t, large_w, 1)
slope, intercept = coeffs

# 估计参数
A_guess = -intercept
tau_guess = A_guess / slope

# 手动调整参数
A_final = A_guess * 1.1
tau_final = tau_guess * 0.9

# 绘制结果
t_model = np.linspace(0, 10, 100)
w_model = W(t_model, A_final, tau_final)

plt.scatter(t_filtered, w_filtered, marker='+', label='Experimental Data')
plt.plot(t_model, w_model, '--', label=f'W(t), A={A_final:.1f}, τ={tau_final:.1f}')
plt.xlabel('Time (hours)')
plt.ylabel('W(t)')
plt.legend()
plt.show()
