import numpy as np
import matplotlib.pyplot as plt

def weierstrass(x, a=0.5, b=3, n_terms=20):
    total = np.zeros_like(x)
    for n in range(n_terms):
        total += a**n * np.cos(b**n * np.pi * x)
    return total

x = np.linspace(-2, 2, 10000)
y = weierstrass(x)
plt.plot(x, y, linewidth=0.5)
plt.title("Weierstrass Function Approximation (n=20)")
plt.show()