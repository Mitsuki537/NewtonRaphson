import numpy as np
import sympy as sp
import tkinter as tk
from tkinter import ttk, messagebox
from sympy.parsing.sympy_parser import parse_expr

class NewtonRaphson3D:
    def __init__(self, root):
        self.root = root
        self.root.title("Newton-Raphson 3D (x, y, z) con Teclado Matemático")
        self.root.geometry("1300x800")
        
        self.vars = ['x', 'y', 'z']
        self.equations = [tk.StringVar() for _ in range(3)]
        self.initial_guesses = {var: tk.DoubleVar(value=1.0) for var in self.vars}
        self.tolerance = tk.DoubleVar(value=0.0001)
        self.max_iterations = tk.IntVar(value=50)
        self.active_entry = None
        self.history = []
        
        self.setup_styles()
        
        self.setup_ui()
        
        self.setup_bindings()
    
    def setup_styles(self):
        """Configura los estilos visuales."""
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('TEntry', padding=5)
        style.configure('Treeview', font=('Arial', 9), rowheight=25)
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        
    def setup_ui(self):
        """Configura la interfaz gráfica."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame, width=450)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        eq_frame = ttk.LabelFrame(left_frame, text="Sistema de Ecuaciones", padding="10")
        eq_frame.pack(fill=tk.X, pady=(0, 10))
        
        for i in range(3):
            ttk.Label(eq_frame, text=f"f{i+1}(x,y,z) =").grid(row=i, column=0, sticky='e', padx=5, pady=2)
            entry = ttk.Entry(eq_frame, textvariable=self.equations[i], width=35, font=('Arial', 10))
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.bind("<FocusIn>", lambda e, idx=i: self.set_active_entry(f"f{idx+1}"))
        
        param_frame = ttk.LabelFrame(left_frame, text="Parámetros", padding="10")
        param_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(param_frame, text="Valores Iniciales:").grid(row=0, column=0, columnspan=2, pady=5)
        
        for i, var in enumerate(self.vars):
            ttk.Label(param_frame, text=f"{var}₀ =").grid(row=i+1, column=0, sticky='e', padx=5, pady=2)
            entry = ttk.Entry(param_frame, textvariable=self.initial_guesses[var], width=12, font=('Arial', 10))
            entry.grid(row=i+1, column=1, sticky='w', pady=2)
            entry.bind("<FocusIn>", lambda e, v=var: self.set_active_entry(v))
        
        ttk.Label(param_frame, text="Tolerancia (% error):").grid(row=4, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(param_frame, textvariable=self.tolerance, width=12, font=('Arial', 10)).grid(row=4, column=1, sticky='w', pady=2)
        
        ttk.Label(param_frame, text="Máx. Iteraciones:").grid(row=5, column=0, sticky='e', padx=5, pady=2)
        ttk.Entry(param_frame, textvariable=self.max_iterations, width=12, font=('Arial', 10)).grid(row=5, column=1, sticky='w', pady=2)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Resolver", command=self.solve, style='TButton').pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(btn_frame, text="Limpiar", command=self.clear_all, style='TButton').pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Button(btn_frame, text="Ejemplo", command=self.load_example, style='TButton').pack(side=tk.LEFT, expand=True, padx=5)
        
        self.setup_math_keyboard(left_frame)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        results_frame = ttk.LabelFrame(right_frame, text="Resultados", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = [("Iter", "Iteración", 70),
                  ("x", "x", 120),
                  ("y", "y", 120),
                  ("z", "z", 120),
                  ("Error", "Error %", 100)]
        
        self.tree = ttk.Treeview(results_frame, columns=[col[0] for col in columns], show="headings")
        
        for col_id, heading, width in columns:
            self.tree.heading(col_id, text=heading, anchor=tk.CENTER)
            self.tree.column(col_id, width=width, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.status_var = tk.StringVar(value="Listo para resolver")
        status_label = ttk.Label(right_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def setup_math_keyboard(self, parent):
        """Crea un teclado matemático completo con pestañas."""
        keyboard_frame = ttk.LabelFrame(parent, text="Teclado Matemático", padding="10")
        keyboard_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        notebook = ttk.Notebook(keyboard_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        num_tab = ttk.Frame(notebook)
        notebook.add(num_tab, text="Básico")
        
        num_layout = [
            ['7', '8', '9', '+', '(', ')'],
            ['4', '5', '6', '-', '^', '**'],
            ['1', '2', '3', '*', 'x', 'y'],
            ['0', '.', '=', '/', 'z', 'π']
        ]
        
        for row in num_layout:
            frame = ttk.Frame(num_tab)
            frame.pack(fill=tk.X)
            for btn in row:
                if btn.strip():
                    ttk.Button(
                        frame, text=btn, width=5,
                        command=lambda b=btn: self.insert_symbol(b),
                        style='TButton'
                    ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)
        
        func_tab = ttk.Frame(notebook)
        notebook.add(func_tab, text="Funciones")
        
        func_layout = [
            ['sin', 'cos', 'tan', 'sqrt'],
            ['asin', 'acos', 'atan', 'exp'],
            ['log', 'ln', 'abs', 'fact']
        ]
        
        for row in func_layout:
            frame = ttk.Frame(func_tab)
            frame.pack(fill=tk.X)
            for btn in row:
                ttk.Button(
                    frame, text=btn, width=8,
                    command=lambda b=btn: self.insert_symbol(f"{b}("),
                    style='TButton'
                ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)
        
        const_tab = ttk.Frame(notebook)
        notebook.add(const_tab, text="Constantes")
        
        const_layout = [
            ['π', 'e', '∞', '←'],
            ['(', ')', '[', ']'],
            ['{', '}', '<', '>']
        ]
        
        for row in const_layout:
            frame = ttk.Frame(const_tab)
            frame.pack(fill=tk.X)
            for btn in row:
                action = "DEL" if btn == '←' else btn
                ttk.Button(
                    frame, text=btn, width=5,
                    command=lambda b=action: self.insert_symbol(b),
                    style='TButton'
                ).pack(side=tk.LEFT, expand=True, padx=2, pady=2)
    
    def setup_bindings(self):
        """Configura los eventos del teclado."""
        self.root.bind("<Control-c>", lambda e: self.clear_all())
        self.root.bind("<Control-r>", lambda e: self.solve())
        self.root.bind("<Control-e>", lambda e: self.load_example())
    
    def set_active_entry(self, entry_name):
        """Establece la entrada activa actual."""
        self.active_entry = entry_name
    
    def insert_symbol(self, symbol):
        """Inserta un símbolo en la entrada activa."""
        if not self.active_entry:
            return
        
        symbol_map = {
            '^': '**',
            'π': 'pi',
            'e': 'exp(1)',
            '∞': 'inf',
            '←': 'DEL',
            'DEL': 'DEL'
        }
        
        symbol = symbol_map.get(symbol, symbol)
        
        if symbol == 'DEL':
            self.delete_last_character()
            return
        
        if self.active_entry.startswith('f'):
            idx = int(self.active_entry[1]) - 1
            current = self.equations[idx].get()
            self.equations[idx].set(current + symbol)
        elif self.active_entry in self.vars:
            current = self.initial_guesses[self.active_entry].get()
            self.initial_guesses[self.active_entry].set(current + symbol)
    
    def delete_last_character(self):
        """Elimina el último carácter de la entrada activa."""
        if not self.active_entry:
            return
            
        if self.active_entry.startswith('f'):
            idx = int(self.active_entry[1]) - 1
            current = self.equations[idx].get()
            self.equations[idx].set(current[:-1])
        elif self.active_entry in self.vars:
            current = self.initial_guesses[self.active_entry].get()
            self.initial_guesses[self.active_entry].set(current[:-1])
    
    def validate_inputs(self):
        """Valida que las ecuaciones y valores iniciales sean correctos."""
        try:
            for eq in self.equations:
                if not eq.get().strip():
                    raise ValueError("Todas las ecuaciones deben estar completas")
                
                x, y, z = sp.symbols('x y z')
                test_expr = parse_expr(eq.get().replace('^', '**'))
                _ = sp.lambdify((x, y, z), test_expr, 'numpy')
                
            for var in self.vars:
                float(self.initial_guesses[var].get())
                
            return True
        except Exception as e:
            messagebox.showerror("Error de validación", f"Entrada inválida:\n{str(e)}")
            return False

    def solve(self):
        """Ejecuta el método de Newton-Raphson para el sistema."""
        if not self.validate_inputs():
            return
    
        try:
            if not all(eq.get().strip() for eq in self.equations):
                raise ValueError("Debe ingresar las 3 ecuaciones")
            
            x, y, z = sp.symbols('x y z')
            eqs = [parse_expr(eq.get().replace('^', '**')) for eq in self.equations]
            F = sp.Matrix(eqs)
            J = F.jacobian([x, y, z])
            
            F_num = sp.lambdify((x, y, z), F, 'numpy')
            J_num = sp.lambdify((x, y, z), J, 'numpy')
            
            X0 = np.array([self.initial_guesses[var].get() for var in self.vars], dtype=float)
            tol = self.tolerance.get() / 100  
            max_iter = self.max_iterations.get()
            
          
            for row in self.tree.get_children():
                self.tree.delete(row)
            
            self.status_var.set("Calculando...")
            self.root.update()
            

            X = X0.copy()
            self.history = []
            
            for iteration in range(max_iter):
                F_val = np.array(F_num(*X)).flatten()
                J_val = J_num(*X)
                
                try:
                    delta = np.linalg.solve(J_val, -F_val)
                except np.linalg.LinAlgError:
                    messagebox.showerror("Error", "Jacobiano singular. No se puede resolver.")
                    self.status_var.set("Error: Jacobiano singular")
                    return
                
                X_new = X + delta
                error = np.linalg.norm(delta) / np.linalg.norm(X_new)
                
                self.history.append({
                    'iter': iteration + 1,
                    'x': X[0], 'y': X[1], 'z': X[2],
                    'error': error * 100
                })
                
                self.tree.insert("", "end", values=(
                    iteration + 1,
                    f"{X[0]:.8f}",
                    f"{X[1]:.8f}",
                    f"{X[2]:.8f}",
                    f"{error * 100:.6f}%"
                ))
                
                if error < tol:
                    self.status_var.set(f"Convergencia en {iteration+1} iteraciones (Error={error*100:.4f}%)")
                    return
                
                X = X_new
            
            self.status_var.set(f"Máximo de iteraciones alcanzado (Error={error*100:.4f}%)")
        
        except np.linalg.LinAlgError:
            self.status_var.set("Error: Jacobiano singular - pruebe otros valores iniciales")
            messagebox.showerror("Error", "El Jacobiano es singular. Intente con otros valores iniciales.")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Ocurrió un error inesperado:\n{str(e)}")
    
    def verify_solution(self):
        """Evalúa las ecuaciones con la solución encontrada"""
        if not self.history:
            messagebox.showwarning("Verificación", "Primero resuelva el sistema")
            return
            
        last_sol = self.history[-1]
        x, y, z = last_sol['x'], last_sol['y'], last_sol['z']
        
        try:
            x_sym, y_sym, z_sym = sp.symbols('x y z')
            residuals = []
            for i, eq in enumerate(self.equations):
                expr = parse_expr(eq.get().replace('^', '**'))
                func = sp.lambdify((x_sym, y_sym, z_sym), expr, 'numpy')
                residuals.append(func(x, y, z))
            
            message = "Residuos de las ecuaciones:\n"
            for i, res in enumerate(residuals):
                message += f"f{i+1} = {res:.2e}\n"
                
            messagebox.showinfo("Verificación de solución", message)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo verificar:\n{str(e)}")


    def clear_all(self):
        """Limpia todos los campos de entrada y resultados."""
        for eq in self.equations:
            eq.set("")
        
        for var in self.vars:
            self.initial_guesses[var].set(1.0)
        
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        self.status_var.set("Campos limpiados")
        self.history = []
    
    def load_example(self):
        """Carga un ejemplo predefinido."""
        self.clear_all()
        
        self.equations[0].set("3*x - cos(y*z) - 1/2")
        self.equations[1].set("x**2 - 81*(y + 0.1)**2 + sin(z + 1.06)")
        self.equations[2].set("exp(-x*y) + 20*z + (10*pi/3) - 1")
        self.initial_guesses['x'].set(0.5)
        self.initial_guesses['y'].set(0.5)
        self.initial_guesses['z'].set(-1)
        
        self.status_var.set("Ejemplo cargado")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonRaphson3D(root)
    root.mainloop()