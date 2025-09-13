import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# --------------------------
# 1. 参数设置 (需根据实际问题调整)
# --------------------------
params = {
    # 常量参数
    'A': 1.0,
    'Q1': 1.0,        # 示例值
    'Q2': 0.5,        # 示例值
    'C': 0.3,         # 示例值
    'lambda1': 2,   # 示例值
    'lambda2': 2,   # 示例值
    'pi': np.pi,      # 圆周率
    'e': np.exp(1),   # 自然常数
    'R1': -2.0,       # 确保a(t)<0的示例值
    'R2': 1.0,        # 确保d(t)>0的示例值
    'D1': 0.1,        # 示例值
    'D2': 0.2,        # 示例值
    'B1': 0.05,       # 示例值
    'B2': 0.08,       # 示例值
    'T': 10.0         # 终端时间
     #初始时间设为 0.0
}
# --------------------------
# 2. 定义积分项的被积函数
# --------------------------
def E(w):
    return np.exp(-0.5 * w)  # 示例：指数衰减函数

def F1(w):
    return w * np.exp(-w) 

def F2(w):
    return np.exp(-2 * w) 

def nu_density(w):
    return np.exp(-w)

# 计算积分项（使用数值积分处理无穷区间）
def compute_integrals():
    # 积分上限（截断无穷区间，根据收敛性调整）
    upper_limit = 100.0
    # 积分1: ∫E(w)²ν(dw)
    def integrand1(w):
        return E(w)**2 * nu_density(w)
    integral1, _ = quad(integrand1, 0, upper_limit)
    
    # 积分2: ∫E(w)F1(w)ν(dw)
    def integrand2(w):
        return E(w) * F1(w) * nu_density(w)
    integral2, _ = quad(integrand2, 0, upper_limit)
    
    # 积分3: ∫F1(w)F2(w)ν(dw)
    def integrand3(w):
        return F1(w) * F2(w) * nu_density(w)
    integral3, _ = quad(integrand3, 0, upper_limit)
    
    # 积分4: ∫F1(w)²ν(dw)
    def integrand4(w):
        return F1(w)**2 * nu_density(w)
    integral4, _ = quad(integrand4, 0, upper_limit)
    
    # 积分5: ∫F2(w)²ν(dw)
    def integrand5(w):
        return F2(w)** 2 * nu_density(w)
    integral5, _ = quad(integrand5, 0, upper_limit)
    
    # 积分6: ∫E(w)F2(w)ν(dw)
    def integrand6(w):
        return E(w) * F2(w) * nu_density(w)
    integral6, _ = quad(integrand6, 0, upper_limit)

    return {
        'int_E2': integral1,
        'int_EF1': integral2,
        'int_F1F2': integral3,
        'int_F12': integral4,
        'int_F22': integral5,
        'int_EF2': integral6
    }

# 预计算积分项，等待调用
integrals = compute_integrals()

# --------------------------
# 3. 定义微分方程组及辅助函数
# --------------------------
def compute_coefficients(P, params, integrals):
    """计算系数a(t), b(t), c(t), d(t), e(t)"""
    R1 = params['R1']
    R2 = params['R2']
    D1 = params['D1']
    D2 = params['D2']
    B1 = params['B1']
    B2 = params['B2']
    C = params['C']
    
    a = R1 + 0.5 * P * (D1**2 + integrals['int_F12'])
    b = P * (B1 + C*D1 + integrals['int_EF1'])
    c = P * (D1*D2 + integrals['int_F1F2'])
    d = R2 + 0.5 * P * (D2**2 + integrals['int_F22'])
    e_term = P * (B2 + C*D2 + integrals['int_EF2'])  # 避免与自然常数e冲突命名为e_term
    
    return a, b, c, d, e_term

#定义微分方程组
def system(t, y, params, integrals):
    """
    定义微分方程组 dy/dt = f(t, y)
    y = [P, S]
    """
    P, S = y
    A = params['A']
    Q1 = params['Q1']
    C = params['C']
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    pi = params['pi']
    e = params['e']
    
    # 计算系数
    a, b, c, d, e_term = compute_coefficients(P, params, integrals)
    
    # 确保满足a<0和d>0的条件
    if a >= 0 or d <= 0:
        raise ValueError(f"Does not satisfy the condition:a={a:.4f} (need<0), d={d:.4f} (need>0)")
    
    # 计算P的导数
    term1 = Q1 + P * A
    term2 = 0.5 * P * (C**2 + integrals['int_E2'])
    numerator = a * e_term**2 + b**2 * d - b * c * e_term
    denominator = 4 * a * d - c**2
    term3 = -numerator / denominator
    dP_dt = term1 + term2 + term3
    
    # 计算S的导数
    log_arg1 = - (pi * e * lambda1) / a  # a<0确保对数内为正
    log_arg2 = (pi * e * lambda2) / d    # d>0确保对数内为正
    dS_dt = (lambda2 - lambda1)/2 + (lambda1/2)*np.log(log_arg1) - (lambda2/2)*np.log(log_arg2)
    
    return [dP_dt, dS_dt]

# --------------------------
# 4. 求解微分方程组
# --------------------------
def solve_ps_system(t_span=None, t_eval=None):
    """求解P(t)和S(t)方程组"""
    # 默认时间设置
    T = params['T']
    if t_span is None:
        t_span = (T, 0.0)  # 反向积分（从终端到初始）
    if t_eval is None:
        t_eval = np.linspace(T, 0.0, 200)  # 评估点
    
    # 终端条件 (t=T时)
    y0 = [2 * params['Q2'], 0.0]  # [P(T), S(T)]
    
    # 求解ODE
    solution = solve_ivp(
        system, 
        t_span, 
        y0, 
        t_eval=t_eval,
        args=(params, integrals),
        method='RK45',  # 可根据问题特性更换为'Radau'等方法
        rtol=1e-6,      # 相对容差 tolerance
    )
    
    return solution

# --------------------------
# 5. 定义函数：计算V(t,x)、π*和γ*
# --------------------------
def compute_V(t_idx, x, solution, params):
    """计算V(t, x) = (1/2)P(t)x² + S(t)"""
    P = solution.y[0, t_idx]
    S = solution.y[1, t_idx]
    return 0.5 * P * x**2 + S

def compute_pi_gamma_stats(t_idx, x, solution, params, integrals):
    """计算π*和γ*的均值和方差"""
    P = solution.y[0, t_idx]
    a, b, c, d, e_term = compute_coefficients(P, params, integrals)
    denominator = 4 * a * d - c**2
    
    # π*的均值和方差
    pi_mean = (c * e_term - 2 * b * d) / denominator * x
    pi_var = -params['lambda1'] / (2 * a)  # a<0确保方差为正
    
    # γ*的均值和方差
    gamma_mean = (b * c - 2 * a * e_term) / denominator * x
    gamma_var = params['lambda2'] / (2 * d)  # d>0确保方差为正
    
    return {
        'pi': {'mean': pi_mean, 'var': pi_var, 'std': np.sqrt(pi_var)},
        'gamma': {'mean': gamma_mean, 'var': gamma_var, 'std': np.sqrt(gamma_var)}
    }

def normal_pdf(x, mean, std):
    """正态分布概率密度函数"""
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)** 2)

# --------------------------
# 6. 可视化函数
# --------------------------
def plot_V(solution):
    """绘制V(t, x)的三维曲面图"""
    # 创建时间和x的网格
    t = solution.t
    x = np.linspace(-10, 10, 100)  # x的取值范围，可根据实际情况调整
    T_grid, X_grid = np.meshgrid(t, x)
    
    # 计算V(t, x)
    V_grid = np.zeros_like(T_grid)
    for i in range(len(t)):
        V_grid[:, i] = compute_V(i, x, solution, params)
    
    # 绘制三维曲面
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_grid, X_grid, V_grid, cmap=cm.coolwarm, 
                          linewidth=0, antialiased=True, alpha=1)
    
    ax.set_title('3D surface plot of V(t,x)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('V(t,x)')
    fig.colorbar(surf, shrink=0.5, aspect=4)
    plt.tight_layout()
    return fig

def plot_pdf_examples(solution, integrals, t_idx=None, x_values=None):
    """绘制特定时间和x值下π*和γ*的概率密度函数"""
    if t_idx is None:
        t_idx = int(len(solution.t)) // 2  # 中间时间点
    if x_values is None:
        x_values = [-5, 0, 5]  # 选取几个典型x值
    
    t_val = solution.t[t_idx]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 为每个分布生成取值范围
    pi_range = np.linspace(-7, 7, 200)
    gamma_range = np.linspace(-7, 7, 200)
    
    # 绘制π*的概率密度
    ax1 = axes[0]
    for x in x_values:
        stats = compute_pi_gamma_stats(t_idx, x, solution, params, integrals)
        pi_pdf = normal_pdf(pi_range, stats['pi']['mean'], stats['pi']['std'])
        ax1.plot(pi_range, pi_pdf, label=f'x={x}',linewidth=2.5)
    ax1.set_title(f'π* probability density function at t={t_val:.2f}')
    ax1.set_xlabel('π')
    ax1.set_ylabel('Probability density')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制γ*的概率密度
    ax2 = axes[1]
    for x in x_values:
        stats = compute_pi_gamma_stats(t_idx, x, solution, params, integrals)
        gamma_pdf = normal_pdf(gamma_range, stats['gamma']['mean'], stats['gamma']['std'])
        ax2.plot(gamma_range, gamma_pdf, label=f'x={x}',linewidth=2.5)
    ax2.set_title(f'γ* probability density function at t={t_val:.2f}')
    ax2.set_xlabel('γ')
    ax2.set_ylabel('Probability density')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

# --------------------------
# 7. 运行求解并可视化所有结果
# --------------------------
if __name__ == "__main__":
    # 求解方程组
    solution = solve_ps_system()
    
    # 提取结果
     # 提取结果
    t = solution.t
    P = solution.y[0]
    S = solution.y[1]
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    
    ax1.plot(t, P, color="#d62728",linestyle='-', linewidth=2.5)
    ax1.set_title('The numerical solution of P(t)')
    ax1.grid(True)
    ax1.set_xlabel('t')
    ax1.set_ylabel('P(t)')
    
    ax2.plot(t, S, color="#000000",linestyle='--', linewidth=2.5)
    ax2.set_title('The numerical solution of S(t)')
    ax2.grid(True)
    ax2.set_xlabel('t')
    ax2.set_ylabel('S(t)')
    
    plt.tight_layout()
    plt.show()
    
    # 绘制V(t, x)的三维图像
    fig_v = plot_V(solution)
    plt.show()
    
    # 绘制特定点的概率密度函数
    fig_pdf = plot_pdf_examples(solution, integrals)
    plt.show()
    