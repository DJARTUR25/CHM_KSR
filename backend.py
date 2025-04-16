# импорт библиотек и функций
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import imageio

"""
Нестационарное уравнение теплопроводности:
    u_t = 9*u_xx + 5*sin(t), x = [0,1]; t = [0,1000]
    НУ:
        u(x, 0) = 1 - x^2
    ГУ:
        u_x(0,t)=0
        u_x(1,t)=7*(u(1,t)-2/7)
"""

class DataContainer:
    def __init__(self):
        self.x = None
        self.t = None
        self.U = None

data = DataContainer()

# основная функция программы, backend
def solve_heat_equation(n, m):
    L = 1.0         # длина стержня
    T = 10.0      # время моделирования
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

    # коэффициенты альфа и бета для метода прогонки
    alpha = np.zeros(n + 1)
    beta  = np.zeros(n + 1)

    # цикл временных итераций
    for j in range(1, m + 1):
        alpha[1] = 1.0
        beta[1]  = 0.0

        for i in range(1, n):
            phi = 5.0 * np.sin(t[j]) + u[i, j-1] / tau
            if abs(C - A * alpha[i]) < 1e-15:
                raise ValueError("Деление на слишком малое число в прямом ходе прогонки.")
            
            alpha[i+1] = B / (C - A * alpha[i])
            beta[i+1]  = (phi + A * beta[i]) / (C - A * alpha[i])

        u[n, j] = (beta[n] + 2*h) / (1 + 7*h - alpha[n])

        for i in range(n-1, -1, -1):
            u[i, j] = alpha[i+1] * u[i+1, j] + beta[i+1]

        u[0, j] = u[1, j]

    return x, t, u

def calculate_error(u_main, u_control, n_main, m_main):
    total_error = 0.0
    max_error = 0.0
    max_error_layer = 0
    count = 0
    for j in range(m_main + 1):
        # Исправление: множитель для времени стал 2 вместо 4
        j_control = j * 2  # Было j * 4
        
        # Проверка выхода за границы
        if j_control >= u_control.shape[1]:
            continue
            
        for i in range(n_main + 1):
            i_control = i * 2
            if i_control >= u_control.shape[0]:
                continue
                
            current_error = abs(u_main[i, j] - u_control[i_control, j_control])
            total_error += current_error
            if current_error > max_error:
                max_error = current_error
                max_error_layer = j
            count += 1
    avg_error = total_error / count if count > 0 else 0.0
    return avg_error, max_error, max_error_layer

def plot_results(ax1, ax2, cax, x, t, u, j, show_control):
    ax1.cla()
    ax2.cla()
    cax.cla()

    X, T_grid = np.meshgrid(x, t)
    mesh = ax1.pcolormesh(X, T_grid, u.T, shading='auto', cmap='viridis')
    ax1.set_title('Распределение температуры', fontsize=10)
    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('t', fontsize=8)

    cb = ax1.figure.colorbar(mesh, cax=cax)
    cb.set_label('u(x,t)', fontsize=8)

    if j >= len(t):
        j = len(t) - 1
    ax2.plot(x, u[:, j], 'r-', label=f't = {t[j]:.2f}')
    ax2.set_title(f'Слой {j}', fontsize=10)
    ax2.set_xlabel('x', fontsize=8)
    ax2.set_ylabel(f'u(x,t)', fontsize=8)
    ax2.legend(fontsize=8)
    plt.subplots_adjust(hspace=0.5)    # Добавить после построения графика:
    ax2.set_ylim(np.min(u)-0.1, np.max(u)+0.1)  # Фиксируем масштаб с небольшим запасом

def create_animation():
    if data.U is None or data.t is None or data.x is None:
        messagebox.showerror("Ошибка", "Сначала выполните расчет!")
        return

    filenames = []
    plt.ioff()
    try:
        for j in range(len(data.t)):
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(data.x, data.U[:, j], 'r-')
            ax.set_ylim(np.min(data.U)-0.1, np.max(data.U)+0.1)
            ax.set_title(f"Слой {j}, t={data.t[j]:.2f}")
            
            frame_path = f"frame_{j:04d}.png"
            fig.savefig(frame_path, dpi=100)
            filenames.append(frame_path)
            plt.close(fig)

        with imageio.get_writer('animation.gif', mode='I', duration=0.1) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in filenames:
            os.remove(filename)

        messagebox.showinfo("Успех", "Анимация создана: animation.gif!")
        
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
    finally:
        plt.ion()

# бомбардиро крокодило

def display_solution(text_widget, x, t, u):
    text_widget.configure(state='normal')
    text_widget.delete("1.0", tk.END)
    
    header = "   t\\x   | " + " ".join(f"{xx:>7.3f}" for xx in x) + "\n"
    text_widget.insert(tk.END, header)
    text_widget.insert(tk.END, "-"*len(header) + "\n")

    for j in range(len(t)):
        line = f"{t[j]:>8.2f} | " + " ".join(f"{u[i,j]:>7.4f}" for i in range(len(x))) + "\n"
        text_widget.insert(tk.END, line)
    
    text_widget.configure(state='disabled')

def on_run(entries, ax1, ax2, cax, canvas, text_widget, check_control, error_vars):
    try:
        n = int(entries['n'].get())
        m = int(entries['m'].get())
        j_input = entries['j'].get()
        t_input = entries['t'].get()
    except ValueError:
        messagebox.showerror("Ошибка", "Некорректный ввод числовых значений")
        return

    x, t, U = solve_heat_equation(n, m)
    data.x, data.t, data.U = x, t, U  # Сохраняем данные
    
    # Определение слоя
    if t_input:
        try:
            t_val = float(t_input)
            j = max(0, np.searchsorted(t, t_val, side='right') - 1)
        except:
            j = 0
    elif j_input:
        j = min(int(j_input), len(t)-1)
    else:
        j = 0

    # Расчет погрешностей
    avg_err, max_err, max_layer = 0.0, 0.0, 0
    if check_control.instate(['selected']):
        n_control = 2 * n
        m_control = 2 * m
        _, _, U_control = solve_heat_equation(n_control, m_control)
        avg_err, max_err, max_layer = calculate_error(U, U_control, n, m)

    error_vars['avg'].set(f"{avg_err:.6f}")
    error_vars['max'].set(f"{max_err:.6f}")
    error_vars['layer'].set(f"{max_layer}")
    
    plot_results(ax1, ax2, cax, x, t, U, j, check_control.instate(['selected']))
    canvas.draw()
    
    text_widget.configure(state='normal')
    text_widget.delete("1.0", tk.END)
    display_solution(text_widget, x, t, U)
    text_widget.configure(state='disabled')
    

def main():
    root = tk.Tk()
    root.title("Теплопроводность")
    mainframe = ttk.Frame(root, padding=10)
    mainframe.pack(fill='both', expand=True)

    left_panel = ttk.Frame(mainframe)
    left_panel.pack(side='left', fill='y')

    entries = {}
    ttk.Label(left_panel, text="Число узлов (n):").grid(row=0, column=0, sticky='w')
    entries['n'] = ttk.Entry(left_panel, width=12)
    entries['n'].grid(row=0, column=1)

    ttk.Label(left_panel, text="Число слоев (m):").grid(row=1, column=0, sticky='w')
    entries['m'] = ttk.Entry(left_panel, width=12)
    entries['m'].grid(row=1, column=1)

    ttk.Label(left_panel, text="Номер слоя (j):").grid(row=2, column=0, sticky='w')
    entries['j'] = ttk.Entry(left_panel, width=12)
    entries['j'].grid(row=2, column=1)

    ttk.Label(left_panel, text="Время (t):").grid(row=3, column=0, sticky='w')
    entries['t'] = ttk.Entry(left_panel, width=12)
    entries['t'].grid(row=3, column=1)

    check_control = ttk.Checkbutton(left_panel, text="Контрольная сетка")
    check_control.grid(row=4, columnspan=2, pady=5)

    error_vars = {
        'avg': tk.StringVar(),
        'max': tk.StringVar(),
        'layer': tk.StringVar()
    }
    
    right_panel = ttk.Frame(mainframe)
    right_panel.pack(side='right', fill='y', padx=10)
    ttk.Label(right_panel, text="Средняя погрешность:").grid(row=0, column=0, sticky='w')
    ttk.Label(right_panel, textvariable=error_vars['avg']).grid(row=0, column=1)
    ttk.Label(right_panel, text="Максимальная погрешность:").grid(row=1, column=0, sticky='w')
    ttk.Label(right_panel, textvariable=error_vars['max']).grid(row=1, column=1)
    ttk.Label(right_panel, text="Слой с максимумом:").grid(row=2, column=0, sticky='w')
    ttk.Label(right_panel, textvariable=error_vars['layer']).grid(row=2, column=1)

    fig = plt.Figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    cax = fig.add_axes([0.91, 0.3, 0.02, 0.4])
    canvas = FigureCanvasTkAgg(fig, master=mainframe)
    canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    text_widget = tk.Text(mainframe, wrap='none', height=15, width=50)
    text_widget.pack(side='bottom', fill='both', expand=True)
    scroll = tk.Scrollbar(mainframe, command=text_widget.yview)
    scroll.pack(side='right', fill='y')
    text_widget.configure(yscrollcommand=scroll.set)

    ttk.Button(left_panel, text="Рассчитать", 
              command=lambda: on_run(entries, ax1, ax2, cax, canvas, text_widget, check_control, error_vars)
              ).grid(row=5, columnspan=2, pady=10)

    ttk.Button(left_panel, text="Создать анимацию", command=create_animation).grid(row=6, columnspan=2, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()