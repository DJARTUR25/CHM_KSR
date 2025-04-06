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

# функция построения графиков
def plot_results(ax1, ax2, cax, x, t, u, t0):
    # очистка осей перед новой отрисовкой
    ax1.cla()
    ax2.cla()
    cax.cla()

    # градиентный график u(t,x)
    X, T_grid = np.meshgrid(x, t)
    mesh = ax1.pcolormesh(X, T_grid, u.T, shading='auto', cmap='viridis')
    ax1.set_title('Распределение температуры')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')

    cb = ax1.figure.colorbar(mesh, cax=cax)
    cb.set_label('u(x,t)')

    # построение графика среза при заданном t0
    if t0 > t[-1]:
        t0 = t[-1]
    idx_t0 = np.argmin(np.abs(t - t0))
    ax2.plot(x, u[:, idx_t0], 'r-', label=f't = {t[idx_t0]:.2f}')
    ax2.set_title('Срез при t0')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t0)')
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
def on_run(entry_n, entry_m, entry_t0, ax1, ax2, cax, canvas, text_widget):
    try:
        n = int(entry_n.get())
        m = int(entry_m.get())
        t0 = float(entry_t0.get())
        if n <= 0 or m <= 0:
            raise ValueError
    except ValueError:      # обработка неправильного ввода
        messagebox.showerror("Ошибка ввода",
                             "n и m — положительные целые; t0 — вещественное число.")
        return

    # Запуск решения
    x, t, U = solve_heat_equation(n, m)
    if x is None or t is None or U is None:
        return
    
    # Построение графиков
    plot_results(ax1, ax2, cax, x, t, U, t0)
    canvas.draw()

    # Вывод в текстовое поле
    display_solution(text_widget, x, t, U)

# main, функция создаёт интерфейс (новое окно, кнопки, поля ввода и т.д.)
def main():
    root = tk.Tk()
    root.title("10(4): Нестационарное уравнение теплопроводности")

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.pack(side='top', fill='both', expand=True)

    # Поля ввода (n, m, t0)
    ttk.Label(mainframe, text="Число узлов (n):").grid(row=0, column=0, sticky='e')
    entry_n = ttk.Entry(mainframe, width=15)
    entry_n.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(mainframe, text="Число слоёв (m):").grid(row=1, column=0, sticky='e')
    entry_m = ttk.Entry(mainframe, width=15)
    entry_m.grid(row=1, column=1, padx=5, pady=5)

    ttk.Label(mainframe, text="Момент времени (t0):").grid(row=2, column=0, sticky='e')
    entry_t0 = ttk.Entry(mainframe, width=15)
    entry_t0.grid(row=2, column=1, padx=5, pady=5)

    display_frame = ttk.Frame(mainframe)
    display_frame.grid(row=4, column=0, columnspan=2, sticky='nsew', pady=10)

    fig = plt.Figure(figsize=(7, 9))
    ax1 = fig.add_axes([0.10, 0.55, 0.70, 0.40])
    ax2 = fig.add_axes([0.10, 0.08, 0.70, 0.35])
    cax = fig.add_axes([0.82, 0.55, 0.04, 0.40])

    canvas = FigureCanvasTkAgg(fig, master=display_frame)
    canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

    text_frame = ttk.Frame(display_frame)
    text_frame.pack(side='right', fill='both', expand=True)

    text_widget = tk.Text(text_frame, wrap='none', height=25)
    text_widget.pack(side='left', fill='both', expand=True)
    scroll_y = tk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
    scroll_y.pack(side='right', fill='y')
    text_widget.configure(yscrollcommand=scroll_y.set)

    run_button = ttk.Button(
        mainframe, text="Решить и Построить",
        command=lambda: on_run(
            entry_n, entry_m, entry_t0,
            ax1, ax2, cax, canvas,
            text_widget
        )
    )
    run_button.grid(row=3, column=0, columnspan=2, pady=5)

    mainframe.columnconfigure(1, weight=1)
    mainframe.rowconfigure(4, weight=1)

    root.mainloop()

# запуск программы
if __name__ == "__main__":
    main()