# импорт библиотек и функций
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""
Нестационарное уравнение теплопроводности:
    u_t = 9*u_xx + 5*sin(t), x = [0,1]; t = [0,1000]
    НУ:
        u(x, 0) = 1 - x^2
    ГУ:
        u_x(0,t)=0
        u_x(1,t)=7*(u(1,t)-2/7)
"""


# основная функция программмы, backend
def solve_heat_equation(n, m):
    L = 1.0         # длина стержня
    T = 1000.0      # время моделирования
    h = L / n       # пространственный шаг
    tau = T / m     # временной шаг
    
    # задаем линейные пространства x, t и u
    x = np.linspace(0, L, n + 1)
    t = np.linspace(0, T, m + 1)
    u = np.zeros((n+1, m+1))        # пространство решений

    # начальное условие
    u[:, 0] = 1.0 - x**2

    # коэффициенты матрицы СЛАУ (для метода прогонки)
    A = 9.0 / h**2                  # нижняя диагональ
    B = 9.0 / h**2                  # верхняя диагональ
    C = 18.0 / h**2 + 1.0 / tau     # главная диагональ

    # коэффициенты альфа и бета для метода прогонки. Беру с запасом, так как подсчет начинается с i = 1
    alpha = np.zeros(n + 1)
    beta  = np.zeros(n + 1)

    # цикл временных итераций, т.е. j = 1, ..., m
    for j in range(1, m + 1):

        # метод прогонки для решения СЛАУ

        # Прямой ход прогонки
        alpha[1] = 1.0
        beta[1]  = 0.0

        for i in range(1, n):
            phi = 5.0 * np.sin(t[j]) + u[i, j-1] / float(tau)  # правая часть СЛАУ 
            if abs(C - A * alpha[i]) < 1e-15:           # обработка ошибки деления на ноль при прямом ходе прогонки
                raise ValueError("Деление на слишком малое число в прямом ходе прогонки.")
            
            alpha[i+1] = B / (C - A * alpha[i])         # подсчет коэффициентов альфа
            beta[i+1]  = (phi + A * beta[i]) / (C - A * alpha[i])   # подсчет коэффициентов бета

        u[n, j] = (beta[n] + 2.0*h) / (1 + 7*h - alpha[n])  # учёт правой границы и ее граничного условия

        # Обратный ход прогонки
        for i in range(n-1, -1, -1):
            u[i, j] = alpha[i+1] * u[i+1, j] + beta[i+1]

        u[0, j] = u[1, j] # учет ГУ для левого конца стержня

    return x, t, u

def calculate_error(u_main, u_control, n_main, m_main):
    """Вычисление средней и максимальной погрешности"""
    total_error = 0.0
    max_error = 0.0
    count = 0
    for j in range(0, m_main+1):
        for i in range(0, n_main+1):
            i_control = i * 2
            j_control = j * 4
            current_error = abs(u_main[i, j] - u_control[i_control, j_control])
            total_error += current_error
            if current_error > max_error:
                max_error = current_error
            count += 1
    avg_error = total_error / count if count > 0 else 0.0
    return avg_error, max_error

def plot_results(ax1, ax2, cax, x, t, u, j, show_control):
    ax1.cla()
    ax2.cla()
    cax.cla()

    X, T_grid = np.meshgrid(x, t)
    mesh = ax1.pcolormesh(X, T_grid, u.T, shading='auto', cmap='viridis')
    
    ax1.set_title('Распределение температуры')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')

    cb = ax1.figure.colorbar(mesh, cax=cax)
    cb.set_label('u(x,t)')

    if j >= len(t):
        j = len(t) - 1
    ax2.plot(x, u[:, j], 'r-', label=f'Слой {j}, t = {t[j]:.2f}')
    ax2.set_title(f'График слоя № {j}, t = {t[j]:.2f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel(f'u(x,t{j})')
    ax2.legend()

# функця вывода массива решения u(x,t)
def display_solution(text_widget, x, t, u):
    text_widget.configure(state='normal')
    text_widget.delete("1.0", tk.END)

    text_widget.insert(tk.END, "Массив решений u(x,t):\n\n")
    
    header = "  t \\ x   | " + "  ".join(f"{xx:>8.4f}" for xx in x) + "\n"
    text_widget.insert(tk.END, header)
    text_widget.insert(tk.END, "-" * len(header) + "\n")
    
    for j in range(len(t)):
        line = f"{t[j]:>8.2f} | " + "  ".join(f"{u[i, j]:>8.4f}" for i in range(len(x))) + "\n"
        text_widget.insert(tk.END, line)

    text_widget.configure(state='disabled')

# функция работы интерфейса, его запуска
def on_run(entry_n, entry_m, entry_j, ax1, ax2, cax, canvas, text_widget, 
          check_control, avg_error_var, max_error_var):
    try:
        n = int(entry_n.get())
        m = int(entry_m.get())
        j = int(entry_j.get())
        if n <= 0 or m <= 0 or j < 0 or j > m:
            raise ValueError
    except ValueError:
        messagebox.showerror("Ошибка ввода", "n и m — положительные целые; j — целое от 0 до m.")
        return

    x, t, U = solve_heat_equation(n, m)
    
    avg_error = 0.0
    max_error = 0.0
    show_control = check_control.instate(['selected'])
    
    if show_control:
        n_control = 2 * n
        m_control = 4 * m
        x_control, t_control, U_control = solve_heat_equation(n_control, m_control)
        avg_error, max_error = calculate_error(U, U_control, n, m)

    # Обновление значений погрешности
    avg_error_var.set(f"{avg_error:.6f}" if show_control else "")
    max_error_var.set(f"{max_error:.6f}" if show_control else "")
    
    plot_results(ax1, ax2, cax, x, t, U, j, show_control)
    canvas.draw()
    
    # Вывод данных в текстовое поле
    text_widget.configure(state='normal')
    text_widget.delete("1.0", tk.END)
    display_solution(text_widget, x, t, U)
    text_widget.configure(state='disabled')

# main, функция создаёт интерфейс (новое окно, кнопки, поля ввода и т.д.)
def main():
    root = tk.Tk()
    root.title("10(4): Нестационарное уравнение теплопроводности")
    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.pack(side='top', fill='both', expand=True)

    # Левая панель: элементы управления
    left_panel = ttk.Frame(mainframe)
    left_panel.grid(row=0, column=0, sticky='nsew')

    # Правая панель: статистика погрешности
    right_panel = ttk.Frame(mainframe)
    right_panel.grid(row=0, column=1, sticky='nsew', padx=10)

    # Элементы управления в левой панели
    ttk.Label(left_panel, text="Число узлов (n):").grid(row=0, column=0, sticky='w')
    entry_n = ttk.Entry(left_panel, width=15)
    entry_n.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(left_panel, text="Число слоёв (m):").grid(row=1, column=0, sticky='w')
    entry_m = ttk.Entry(left_panel, width=15)
    entry_m.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(left_panel, text="Номер слоя (j):").grid(row=2, column=0, sticky='w')
    entry_j = ttk.Entry(left_panel, width=15)
    entry_j.grid(row=2, column=1, padx=5, pady=5)

    check_control = ttk.Checkbutton(left_panel, text="Включить контрольную сетку")
    check_control.grid(row=3, column=0, columnspan=2, pady=5)

    run_button = ttk.Button(
        left_panel, text="Решить и Построить",
        command=lambda: on_run(entry_n, entry_m, entry_j, ax1, ax2, cax, canvas, text_widget, check_control,avg_error_var, max_error_var)
    )
    run_button.grid(row=4, column=0, columnspan=2, pady=10)

    # Статистика погрешности в правой панели
    avg_error_var = tk.StringVar()
    max_error_var = tk.StringVar()
    
    ttk.Label(right_panel, text="Средняя погрешность:").grid(row=0, column=0, sticky='w')
    ttk.Label(right_panel, textvariable=avg_error_var).grid(row=0, column=1, sticky='w')
    
    ttk.Label(right_panel, text="Максимальное отклонение:").grid(row=1, column=0, sticky='w')
    ttk.Label(right_panel, textvariable=max_error_var).grid(row=1, column=1, sticky='w')

    # Графики и вывод данных
    display_frame = ttk.Frame(mainframe)
    display_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=10)

    root.mainloop()

# запуск программы
if __name__ == "__main__":
    main()