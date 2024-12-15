import numpy as np

# Параметры
a = 3  # sqrt(9)
T = 1000
L = 1
n = 100
m = 1000
h = L / n
tau = T / m
alpha = a**2 * tau / h**2

# Сетка
x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# Начальное условие
u = np.zeros((m+1, n+1))
u[0, :] = 1 - x**2

def thomas_algorithm(a, b, c, d):
    """
    Решение трехдиагональной системы методом прогонки.
    a: нижняя диагональ (длина n-1),
    b: главная диагональ (длина n),
    c: верхняя диагональ (длина n-1),
    d: правая часть (длина n).
    """
    n = len(b)
    # Прямой ход
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        temp = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / temp
    
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / (b[n-1] - a[n-2] * c_prime[n-2])
    
    # Обратный ход
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


# Граничные условия
def g_left(u1):
    return u1

def g_right(un, h):
    return un * (1 + 7*h) - 2*h

# Матрица трехдиагональной системы
a_diag = -alpha * np.ones(n-2)  # нижняя диагональ
b_diag = (1 + 2*alpha) * np.ones(n-1)  # главная диагональ
c_diag = -alpha * np.ones(n-2)  # верхняя диагональ

# Основной цикл по времени
for j in range(m):
    # Формируем правую часть
    b = u[j, 1:n] + tau * 5 * np.sin(t[j+1])
    
    # Учет граничных условий
    b[0] += alpha * g_left(u[j+1, 1])  # левая граница
    b[-1] += alpha * g_right(u[j+1, n], h)  # правая граница
    
    # Решение методом прогонки
    u_inner = thomas_algorithm(a_diag, b_diag, c_diag, b)
    
    # Заполнение нового слоя
    u[j+1, 1:n] = u_inner
    u[j+1, 0] = g_left(u[j+1, 1])
    u[j+1, n] = g_right(u[j+1, n-1], h)