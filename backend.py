import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ввод размеров сетки
Nx = int(input("Введите количество шагов по пространству (Nx): "))
Nt = int(input("Введите количество шагов по времени (Nt): "))

# Параметры задачи
L = 1  # Длина стержня
T = 1000  # Время
dx = L / Nx
dt = T / Nt

alpha_coef = 9  # Коэффициент температуропроводимости

# Сетки по x и t
x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)

# Начальное распределение температуры
u = np.zeros((Nx + 1, Nt + 1))
u[:, 0] = 1 - x**2

# Коэффициенты для метода прогонки
A_coef = alpha_coef * dt / dx**2
B_coef = A_coef
C_coef = 1 + 2 * A_coef

for j in range(0, Nt):
    # Инициализация массивов alpha и betta
    alpha = np.zeros(Nx + 1)
    betta = np.zeros(Nx + 1)

    # Прямой ход прогонки
    alpha[1] = 0  # Поскольку u[0] = u[1] по условию
    betta[1] = u[0, j]

    for i in range(1, Nx):
        phi = dt * 5 * np.sin(t[j]) + u[i, j]
        denom = C_coef - A_coef * alpha[i]
        alpha[i + 1] = B_coef / denom
        betta[i + 1] = (phi + A_coef * betta[i]) / denom

    # Граничное условие третьего рода (x=L)
    u[Nx, j + 1] = (betta[Nx] + 2 * dx) / (1 + 7 * dx - alpha[Nx])

    # Обратный ход прогонки
    for i in range(Nx - 1, 0, -1):
        u[i, j + 1] = alpha[i + 1] * u[i + 1, j + 1] + betta[i + 1]

    # Граничное условие второго рода (x=0)
    u[0, j + 1] = u[1, j + 1]

# Построение 3D-графика
X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.title('Распределение температуры')
plt.show()