import numpy as np

# parameter values
a = 3
T = 1000
L = 1
n = 100
m = 1000
h = L / n
tau = T / m
gamma = pow(a, 2)*tau/pow(h, 2)

# makes a net
x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# The initial condition
u = np.zeros((m+1, n+1))
u[0, :] = 1 - pow(x,2)

# Bondary conditions
def g_left(u1):
    return u1

def g_right(un_, h):
    return un_ * (1 + 7*h) - 2*h

# Method of running the solution of SLAE
def ttridiag_solver(a, b, c, d):
    N = len(b) # length of array b equal dimension of matrix
    # straight stroke: brings the matrix to an upper-triangular shape
    for i in range(1, N):
        



    # Обратный ход
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x




# Матрица трехдиагональной системы
a_diag = -gamma * np.ones(n-2)  # нижняя диагональ
b_diag = (1 + 2*gamma) * np.ones(n-1)  # главная диагональ
c_diag = -gamma * np.ones(n-2)  # верхняя диагональ

# Основной цикл по времени
for j in range(m):
    # Формируем правую часть
    b = u[j, 1:n] + tau * 5 * np.sin(t[j+1])
    
    # Учет граничных условий
    b[0] += gamma * g_left(u[j+1, 1])  # левая граница  
    b[-1] += gamma * g_right(u[j+1, n], h)  # правая граница
    
    # Решение методом прогонки
    u_inner = thomas_algorithm(a_diag, b_diag, c_diag, b)
    
    # Заполнение нового слоя
    u[j+1, 1:n] = u_inner
    u[j+1, 0] = g_left(u[j+1, 1])
    u[j+1, n] = g_right(u[j+1, n-1], h)