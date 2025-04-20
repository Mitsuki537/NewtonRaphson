import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from math import sin, cos, tan, log, asin, acos, atan, exp, log10, pi, sqrt, factorial
import json
import os
from datetime import datetime

class NewtonRaphsonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Newton-Raphson para Sistemas No Lineales")
        self.root.geometry("1000x700")
        
        # Configuración inicial
        self.history_file = "history.json"
        self.history = self.load_history()
        self.colors = {
            'panel': '#f8f9fa',
            'primary': '#007bff',
            'info': '#17a2b8',
            'background': '#f0f0f0'
        }
        
        self.setup_styles()
        self.setup_ui()
    
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Modern.TFrame', background=self.colors['background'])
        style.configure('Modern.TLabel', background=self.colors['background'])
        style.configure('Modern.TNotebook', background=self.colors['background'])
        style.configure('Modern.TNotebook.Tab', padding=[10, 5], background=self.colors['background'])
    
    def setup_ui(self):
        # Panel principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo (entradas y controles)
        left_frame = ttk.Frame(main_frame, width=350, style='Modern.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Panel derecho (gráfico y resultados)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # --- Sección de entrada de funciones ---
        input_frame = ttk.Frame(left_frame, padding="10", style='Modern.TFrame')
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Sistema de Ecuaciones:", style='Modern.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(input_frame, text="f₁(x, y) =", style='Modern.TLabel').grid(row=1, column=0, sticky='e')
        self.f1_entry = ttk.Entry(input_frame, width=25)
        self.f1_entry.grid(row=1, column=1, sticky='ew')
        self.f1_entry.insert(0, "3*x**2 + 5*y**2 - 20 - 9*y")
        
        ttk.Label(input_frame, text="f₂(x, y) =", style='Modern.TLabel').grid(row=2, column=0, sticky='e')
        self.f2_entry = ttk.Entry(input_frame, width=25)
        self.f2_entry.grid(row=2, column=1, sticky='ew')
        self.f2_entry.insert(0, "4*x**2 - y**2 + 4")
        
        # Valores iniciales
        ttk.Label(input_frame, text="x₀:", style='Modern.TLabel').grid(row=3, column=0, sticky='e')
        self.x0_entry = ttk.Entry(input_frame, width=10)
        self.x0_entry.grid(row=3, column=1, sticky='w')
        self.x0_entry.insert(0, "1.0")
        
        ttk.Label(input_frame, text="y₀:", style='Modern.TLabel').grid(row=4, column=0, sticky='e')
        self.y0_entry = ttk.Entry(input_frame, width=10)
        self.y0_entry.grid(row=4, column=1, sticky='w')
        self.y0_entry.insert(0, "3.0")
        
        # Botones de control
        control_frame = ttk.Frame(input_frame, style='Modern.TFrame')
        control_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(control_frame, text="Calcular", command=self.calcular).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Limpiar", command=self.clear_fields).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Guardar", command=self.save_current).pack(side=tk.LEFT, padx=2)
        
        # --- Teclado matemático ---
        self.create_math_keyboard(left_frame)
        
        # --- Historial de ejercicios ---
        self.setup_history_ui(left_frame)
        
        # --- Sección de gráfico ---
        graph_frame = ttk.Frame(right_frame)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # --- Tabla de iteraciones ---
        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ("Iteración", "x", "y", "Error x (%)", "Error y (%)")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor='center')
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_math_keyboard(self, parent):
        """Crea el teclado matemático con pestañas"""
        keyboard_frame = ttk.Frame(parent, style='Modern.TFrame', padding=(10, 5))
        keyboard_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        notebook = ttk.Notebook(keyboard_frame, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de números
        num_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(num_tab, text="Números")
        
        num_buttons = [
            ['7', '8', '9', '+', '(', ')'],
            ['4', '5', '6', '-', '^', '√'],
            ['1', '2', '3', '*', 'π', 'e'],
            ['0', '.', '=', '/', 'x', '^2']
        ]
        
        for row in num_buttons:
            frame = ttk.Frame(num_tab, style='Modern.TFrame')
            frame.pack(fill=tk.X, pady=2)
            for btn in row:
                btn_widget = tk.Button(
                    frame, text=btn, 
                    bg=self.colors['panel'], 
                    fg=self.colors['primary'],
                    activebackground=self.colors['info'],
                    activeforeground='white',
                    font=('Segoe UI', 10),
                    relief='flat',
                    borderwidth=0,
                    command=lambda b=btn: self.insert_math_symbol(b)
                )
                btn_widget.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, ipady=8)
                
                # Efecto hover
                btn_widget.bind('<Enter>', lambda e, b=btn_widget: b.config(bg='#e8f4ff'))
                btn_widget.bind('<Leave>', lambda e, b=btn_widget: b.config(bg=self.colors['panel']))
        
        # Pestaña de funciones
        func_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(func_tab, text="Funciones")
        
        func_buttons = [
            ['sin', 'cos', 'tan', 'log'],
            ['asin', 'acos', 'atan', 'ln'],
            ['abs', 'exp', 'fact', '√']
        ]
        
        for row in func_buttons:
            frame = ttk.Frame(func_tab, style='Modern.TFrame')
            frame.pack(fill=tk.X, pady=2)
            for btn in row:
                btn_widget = tk.Button(
                    frame, text=btn, 
                    bg=self.colors['panel'], 
                    fg=self.colors['primary'],
                    activebackground=self.colors['info'],
                    activeforeground='white',
                    font=('Segoe UI', 10),
                    relief='flat',
                    borderwidth=0,
                    command=lambda b=btn: self.insert_math_symbol(f"{b}(")
                )
                btn_widget.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2, ipady=8)
                
                # Efecto hover
                btn_widget.bind('<Enter>', lambda e, b=btn_widget: b.config(bg='#e8f4ff'))
                btn_widget.bind('<Leave>', lambda e, b=btn_widget: b.config(bg=self.colors['panel']))
    
    def setup_history_ui(self, parent):
        """Configura la interfaz del historial"""
        history_frame = ttk.Frame(parent, style='Modern.TFrame')
        history_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        ttk.Label(history_frame, text="Historial de Ejercicios", style='Modern.TLabel').pack()
        
        # Lista de historial
        self.history_listbox = tk.Listbox(
            history_frame,
            bg='white',
            relief='flat',
            selectbackground=self.colors['info'],
            selectmode=tk.SINGLE,
            height=8
        )
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botones de historial
        history_btn_frame = ttk.Frame(history_frame, style='Modern.TFrame')
        history_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(history_btn_frame, text="Cargar", command=self.load_selected).pack(side=tk.LEFT, expand=True)
        ttk.Button(history_btn_frame, text="Eliminar", command=self.delete_selected).pack(side=tk.LEFT, expand=True)
        
        self.update_history_list()
    
    def insert_math_symbol(self, symbol):
        """Inserta símbolos matemáticos en la entrada activa"""
        mapping = {
            '^2': '**2',
            '√': 'sqrt(', 
            'π': 'pi',
            '^': '**',
            'fact': 'factorial(',
            'abs': 'abs(',
            'e': 'exp(',
            'sin': 'sin(',
            'cos': 'cos(',
            'tan': 'tan(',
            'log': 'log10(',
            'ln': 'log(',
            'exp': 'exp('
        }
        
        # Determinar qué entrada está activa
        if self.f1_entry == self.root.focus_get():
            entry = self.f1_entry
        elif self.f2_entry == self.root.focus_get():
            entry = self.f2_entry
        else:
            return  # Ninguna entrada activa
        
        # Obtener posición del cursor
        cursor_pos = entry.index(tk.INSERT)
        current_text = entry.get()
        before = current_text[:cursor_pos]
        after = current_text[cursor_pos:]
        
        # Insertar el símbolo mapeado
        text_to_insert = mapping.get(symbol, symbol)
        
        # Caso especial: si insertamos función después de letra, agregar *
        if text_to_insert.endswith('(') and before and before[-1].isalpha():
            text_to_insert = '*' + text_to_insert
        
        entry.delete(0, tk.END)
        entry.insert(0, before + text_to_insert + after)
        
        # Posicionar cursor
        if text_to_insert.endswith('('):
            new_pos = cursor_pos + len(text_to_insert)
            entry.icursor(new_pos)
    
    def calcular(self):
        try:
            # Definir F y J
            f1 = lambda x, y: eval(self.f1_entry.get(), {'x': x, 'y': y, **globals()})
            f2 = lambda x, y: eval(self.f2_entry.get(), {'x': x, 'y': y, **globals()})
            F = lambda X: np.array([f1(X[0], X[1]), f2(X[0], X[1])])
            
            # Jacobiano numérico
            h = 1e-5
            J = lambda X: np.array([
                [(f1(X[0]+h, X[1]) - f1(X[0], X[1]))/h, (f1(X[0], X[1]+h) - f1(X[0], X[1]))/h],
                [(f2(X[0]+h, X[1]) - f2(X[0], X[1]))/h, (f2(X[0], X[1]+h) - f2(X[0], X[1]))/h]
            ])
            
            X0 = [float(self.x0_entry.get()), float(self.y0_entry.get())]
            sol, iterations, history = newton_raphson_system(F, J, X0)
            
            if sol is not None:
                self.actualizar_grafico(F, history)
                self.actualizar_tabla(history)
        except Exception as e:
            messagebox.showerror("Error", f"Revise las ecuaciones:\n{str(e)}")
    
    def actualizar_grafico(self, F, history):
        self.ax.clear()
        
        # Rango del gráfico
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5
        
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluar funciones
        Z1 = eval(self.f1_entry.get(), {'x': X, 'y': Y, **globals()})
        Z2 = eval(self.f2_entry.get(), {'x': X, 'y': Y, **globals()})
        
        # Curvas de nivel
        self.ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        self.ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        
        # Trayectoria de iteraciones
        history = np.array(history)
        self.ax.plot(history[:, 0], history[:, 1], 'ko-', markersize=5, label="Iteraciones")
        self.ax.scatter(history[-1, 0], history[-1, 1], c='green', s=100, marker='*', label="Solución")
        
        # Configuración
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.legend()
        self.ax.grid()
        self.canvas.draw()
    
    def actualizar_tabla(self, history):
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        for i, X in enumerate(history):
            if i == 0:
                error_x = 0.0
                error_y = 0.0
            else:
                error_x = abs((X[0] - history[i-1][0]) / abs(X[0])) * 100 if X[0] != 0 else 0
                error_y = abs((X[1] - history[i-1][1]) / abs(X[1])) * 100 if X[1] != 0 else 0
            
            self.tree.insert("", "end", values=(
                i,
                f"{X[0]:.6f}",
                f"{X[1]:.6f}",
                f"{error_x:.4f}",
                f"{error_y:.4f}"
            ))
    
    def save_current(self):
        """Guarda el ejercicio actual en el historial"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = {
            'timestamp': timestamp,
            'f1': self.f1_entry.get(),
            'f2': self.f2_entry.get(),
            'x0': self.x0_entry.get(),
            'y0': self.y0_entry.get()
        }
        
        self.history.append(entry)
        self.save_history()
        self.update_history_list()
        messagebox.showinfo("Guardado", "Ejercicio guardado en historial")
    
    def load_selected(self):
        """Carga un ejercicio del historial"""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        entry = self.history[index]
        
        self.clear_fields()
        
        self.f1_entry.insert(0, entry['f1'])
        self.f2_entry.insert(0, entry['f2'])
        self.x0_entry.insert(0, entry['x0'])
        self.y0_entry.insert(0, entry['y0'])
    
    def delete_selected(self):
        """Elimina un ejercicio del historial"""
        selection = self.history_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        del self.history[index]
        self.save_history()
        self.update_history_list()
    
    def clear_fields(self):
        """Limpia todos los campos de entrada"""
        self.f1_entry.delete(0, tk.END)
        self.f2_entry.delete(0, tk.END)
        self.x0_entry.delete(0, tk.END)
        self.y0_entry.delete(0, tk.END)
        self.x0_entry.insert(0, "1.0")
        self.y0_entry.insert(0, "1.0")
        
        # Limpiar resultados anteriores
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.ax.clear()
        self.canvas.draw()
    
    def update_history_list(self):
        """Actualiza la lista del historial"""
        self.history_listbox.delete(0, tk.END)
        for entry in self.history:
            display_text = f"{entry['timestamp']}: {entry['f1'][:20]}... y {entry['f2'][:20]}..."
            self.history_listbox.insert(tk.END, display_text)
    
    def load_history(self):
        """Carga el historial desde archivo"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def save_history(self):
        """Guarda el historial en archivo"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

def newton_raphson_system(F, J, X0, tol=1e-6, max_iter=100):
    X = np.array(X0, dtype=float)
    history = [X.copy()]
    
    for iteration in range(max_iter):
        F_X = F(X)
        J_X = J(X)
        
        try:
            delta_X = np.linalg.solve(J_X, -F_X)
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "Jacobiano singular. No se puede resolver.")
            return None, None, None
        
        X_new = X + delta_X
        history.append(X_new.copy())
        
        if np.linalg.norm(X_new - X) < tol:
            return X_new, iteration + 1, history
        
        X = X_new
    
    messagebox.showwarning("Advertencia", f"No convergió en {max_iter} iteraciones.")
    return X, max_iter, history

if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonRaphsonApp(root)
    root.mainloop()