import numpy as np
import tkinter as tk
import matplotlib

# Parametres
a = 3
T = 1000
L = 1
default_n = 100
default_m = 1000

# Bounds
def g_left(u1):
    return u1

def g_right(un_1, h):
    return un_1 * (1 + 7 * h) - 2 * h

# Running method for solving SLAE
def tridiag_solver(A, b):
    N = len(A)
    alpha = np.zeros(N)
    beta = np.zeros(N)

    lower = np.zeros(N-1)
    main = np.zeros(N)
    upper = np.zeros(N-1)

    for i in range (0, N):
        main[i] = A[i][i]
    
    for i in range (0, N-1):
        lower[i] = A[i+1][i]
        upper[i] = A[i][i+1]


    # Normalization of matrix
    upper[0] /= main[0]
    b[0] /= main[0]
    main[0] = 1.

    lower[N-2] /= main[N-1]
    b[N-1] /= main[N-1]
    main[N-1] = 1.

    alpha[1] = -upper[0]
    beta[1] = b[0]
    # forward elimination

    for i in range (0, N-1):
        alpha[i+1] = upper[i] / (-main[i] - alpha[i] * lower[i])
        beta[i+1] = (-b[i] + lower[i] * beta[i]) / (-main[i] - alpha[i] * lower[i])

    y = np.zeros(N)
    y[N-1] = ( b[N-1] + A[N-1][N-2] * beta[N-1]) / (1 + A[N-1][N-2] * alpha[N-1])
    
    print (alpha, beta)

    # back subtitution
    for i in range (N-2, 0, -1):
        y[i] = alpha[i+1] * y[i+1] + beta[i+1]

    #print (y)
    return y

# def main_model(default_n, default_m):
    h = L / default_n
    tau = T / default_m
    gamma = pow(a, 2) * tau / pow(h, 2)

    # a net
    x = np.linspace(0, L, default_n + 1)
    t = np.linspace(0, T, default_m + 1)

    # Initial condition
    u = np.zeros((default_m + 1, default_n + 1))
    u[0, :] = 1 - pow(x, 2)

    # makes a tridiagonal metrix
    lower_diag = -gamma * np.ones(default_n - 1)            # lower diagonal (size: n-1)
    main_diag = (1 / tau + 2 * gamma) * np.ones(default_n)  # main diagonal (size:n)
    upper_diag = -gamma * np.ones(default_n - 1)            # upper diagonal (size: n-1)
    
    for j in range(default_m):
        # makes the right part
        right_part = u[j, 1:default_n] + tau * 5 * np.sin(t[j + 1])
    
        # check the bounds
        right_part[0] += gamma * g_left(u[j + 1, 1])            # left bound
        right_part[-1] += gamma * g_right(u[j + 1, default_n - 1], h)   # right bound
    
        # dolving the system
        u_inner = tridiag_solver(lower_diag, main_diag[1:default_n], upper_diag, right_part)

        # filling in a new layer
        u[j + 1, 1:default_n] = u_inner
        u[j + 1, 0] = g_left(u[j + 1, 1])
        u[j + 1, default_n] = g_right(u[j + 1, default_n - 1], h)

    return u

# def test_tridiag_solver():
    # 1st test: simple case
    A = [[3, 1, 0, 0], [1, 2, 1, 0], [0, 3, 6, 2],[0, 0, 5, 7]]
    b = [4, 4, 11, 12]
    tridiag_solver(A, b)

A = [[3, 1, 0, 0], [1, 2, 1, 0], [0, 3, 6, 2], [0, 0, 5, 7]]
b = [4, 4, 11, 12]
tridiag_solver(A, b)
