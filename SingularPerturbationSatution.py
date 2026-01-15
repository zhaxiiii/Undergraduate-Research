import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

#参数
epsilon = 0.05

#精确解
x_exact = np.linspace(0, 1, 1000)
y_exact = (1 - np.exp(-x_exact/epsilon)) / (1 - np.exp(-1/epsilon))

#奇异摄动近似解
y_approx = 1 - np.exp(-x_exact/epsilon)

#退化方程解
y_reduced = np.ones_like(x_exact)

#数值求解原方程
def ode(x, y):
    return np.vstack((y[1], -y[1]/epsilon))

def bc(ya, yb):
    return np.array([ya[0], yb[0]-1])

x_num = np.linspace(0, 1, 100)
y_num = np.zeros((2, x_num.size))
y_num[0] = x_num  # 初始猜测

sol = solve_bvp(ode, bc, x_num, y_num, max_nodes=10000)
x_sol = sol.x
y_sol = sol.y[0]

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'k-', linewidth=2, label='Exact Solution')
plt.plot(x_exact, y_approx, 'r--', linewidth=2, label='Singular Perturbation Approx')
plt.plot(x_exact, y_reduced, 'b:', linewidth=2, label='Reduced Solution (ε=0)')
plt.plot(x_sol, y_sol, 'go', markersize=4, label='Numerical Solution (solve_bvp)', alpha=0.6)

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title(f'Boundary Layer Behavior: ε = {epsilon}', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.axvline(x=5*epsilon, color='gray', linestyle='--', alpha=0.5, label='~5ε (Boundary Layer Width)')
plt.legend()

# 子图：边界层细节
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
x_detail = np.linspace(0, 5*epsilon, 200)
y_exact_detail = (1 - np.exp(-x_detail/epsilon)) / (1 - np.exp(-1/epsilon))
y_approx_detail = 1 - np.exp(-x_detail/epsilon)
plt.plot(x_detail, y_exact_detail, 'k-', label='Exact')
plt.plot(x_detail, y_approx_detail, 'r--', label='Approx')
plt.xlabel('x (zoomed)')
plt.ylabel('y')
plt.title('Inside Boundary Layer')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
error = np.abs(y_exact - y_approx)
plt.semilogy(x_exact, error, 'b-')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('Approximation Error')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()