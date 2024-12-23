import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Метод прогонки для решения трехдиагональной матрицы
def tridiag_solver(a, b, c, d):

    n = len(d)
    alpha = np.zeros(n-1)
    beta = np.zeros(n)
    

    if abs(b[0]) < 1e-15:   # исключение, если деление на 0
        raise ValueError("Нулевой элемент на главной диагонали (b[0] = 0).")

    # Прямой ход прогонки
    alpha[0] = c[0] / b[0]
    beta[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1]*alpha[i-1]
        if abs(denom) < 1e-15:
            raise ValueError("Деление на ноль при прогонке.")
        alpha[i] = c[i] / denom
        beta[i] = (d[i] - a[i-1]*beta[i-1]) / denom
    
    denom = b[-1] - a[-2]*alpha[-2]
    if abs(denom) < 1e-15:
        raise ValueError("Деление на ноль при прогонке (последняя строка).")
    beta[-1] = (d[-1] - a[-2]*beta[-2]) / denom
    
    # Обратный ход
    x = np.zeros(n)
    x[-1] = beta[-1]
    for i in range(n-2, -1, -1):
        x[i] = beta[i] - alpha[i]*x[i+1]

    # возвращает массив решений СЛАУ
    return x

def solve_heat_equation(n, m):
    """
    Нестационарное уравнение теплопроводности:
        u_t = 9*u_xx + 5*sin(t), x = [0,1]; t = [0,1000]
        ГУ:
        u_x(0,t)=0 
        u_x(1,t)=7*(u(1,t)-2/7)
    """
    L = 1.0         # длина стержня
    T = 1000.0      # время моделирования
    h = L / n       # шаг по длине
    tau = T / m     # шаг по времени
    a_coef = 9.0    # коэффициент температуропроводности
    
    x = np.linspace(0, L, n+1)
    t = np.linspace(0, T, m+1)
    u = np.zeros((n+1, m+1))

    # Начальное условие
    u[:, 0] = 1 - x**2

    A = a_coef * tau / h**2
    
    # запуск цикла: проходим каждый слой, для каждого слоя решаем СЛАУ и рисуем решения
    for j in range(m):
        a_arr = np.full(n-1, -A)         
        b_arr = np.full(n-1, 1 + 2*A)    
        c_arr = np.full(n-1, -A)         
        d_arr = u[1:n, j] + tau*5*np.sin(t[j])
        
        # записываем первый элемент в столбец b
        d_arr[0] += A * u[0, j]
        
        # Решаем
        try:
            u_internal = tridiag_solver(a_arr, b_arr, c_arr, d_arr)
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return None, None, None
        
        # запись решений в отдельный массив
        u[1:n, j+1] = u_internal
        u[0, j+1] = u[1, j+1]

        
        denom = (1 - 7*h)
        if abs(denom) < 1e-15:
            messagebox.showwarning("Предупреждение","(1 - 7*h) близко к нулю!")
        u[n, j+1] = (u[n-1, j+1] - 2*h) / denom
    
    return x, t, u


# функция построения графиков
def plot_results(ax1, ax2, cax, x, t, u, t0):

    # Очищаем то, что было:
    ax1.cla()
    ax2.cla()
    cax.cla()
    
    # 1) Градиентный график u(x,t)
    X, T_grid = np.meshgrid(x, t)
    mesh = ax1.pcolormesh(X, T_grid, u.T, shading='auto', cmap='viridis')
    ax1.set_title('Распределение температуры')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    
    # Связь colorbar именно с осью cax
    cb = ax1.figure.colorbar(mesh, cax=cax)
    cb.set_label('u(x,t)')
    
    # 2) График u(x, t0)
    if t0 > t[-1]:
        t0 = t[-1]
    idx_t0 = np.argmin(np.abs(t - t0))

    ax2.plot(x, u[:, idx_t0], 'r-', label=f't = {t[idx_t0]:.2f}')
    ax2.set_title('Срез при t0')
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t0)')
    ax2.legend()


def display_solution(text_widget, x, t, u):     # вывод решения в текст справа в окне
    text_widget.configure(state='normal')
    text_widget.delete("1.0", tk.END)

    text_widget.insert(tk.END, "Массив решений u(x,t):\n\n")
    
    header = " t \\ x   | " + "  ".join(f"{xx:>8.4f}" for xx in x) + "\n"
    text_widget.insert(tk.END, header)
    text_widget.insert(tk.END, "-"*len(header) + "\n")
    
    for j in range(len(t)):
        line = f"{t[j]:>8.2f} | " + "  ".join(f"{u[i, j]:>8.4f}" for i in range(len(x))) + "\n"
        text_widget.insert(tk.END, line)

    text_widget.configure(state='disabled')

def on_run(entry_n, entry_m, entry_t0, ax1, ax2, cax, canvas, text_widget):     # функция ввода данных и запуска решений и построений графиков
    try:
        n = int(entry_n.get())
        m = int(entry_m.get())
        t0 = float(entry_t0.get())
        if n <= 0 or m <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Ошибка ввода",
            "n и m — положительные целые; t0 — вещественное число.")
        return

    # Решение
    x, t, U = solve_heat_equation(n, m)
    if x is None or t is None or U is None:
        return
    
    # Графики
    plot_results(ax1, ax2, cax, x, t, U, t0)
    canvas.draw()

    # Таблица решений
    display_solution(text_widget, x, t, U)


def main():
    root = tk.Tk()
    root.title("10(4): Нестационарное уравнение теплопроводности")

    mainframe = ttk.Frame(root, padding="10 10 10 10")
    mainframe.pack(side='top', fill='both', expand=True)

    # Поля для ввода
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


if __name__ == "__main__":
    main()