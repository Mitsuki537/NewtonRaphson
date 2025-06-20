import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
from sympy import symbols, sympify, diff, Matrix, lambdify
from sympy.parsing.sympy_parser import (standard_transformations, 
                                      implicit_multiplication, 
                                      convert_xor)
import os

class NewtonRaphsonSystemSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Newton-Raphson Multivariable (5D)")
        self.root.geometry("1400x800")
        
        # Variables y configuración inicial
        self.vars = ['x', 'y', 'z', 'w', 'r']
        self.equations = [tk.StringVar() for _ in range(5)]
        self.initial_values = {var: tk.DoubleVar(value=1.0) for var in self.vars}
        self.error_tolerance = tk.DoubleVar(value=0.0001)
        self.max_iterations = tk.IntVar(value=50)
        
        # Interfaz gráfica
        self.setup_ui()
        self.setup_menu()
    
    def setup_menu(self):
        """Configura el menú superior para guardar/cargar archivos"""
        menubar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Nuevo", command=self.new_file)
        file_menu.add_command(label="Abrir", command=self.open_file)
        file_menu.add_command(label="Guardar", command=self.save_file)
        file_menu.add_command(label="Guardar como...", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)
        
        menubar.add_cascade(label="Archivo", menu=file_menu)
        self.root.config(menu=menubar)
    
    def new_file(self):
        """Limpia todos los campos para un nuevo archivo"""
        for eq in self.equations:
            eq.set("")
        for var in self.vars:
            self.initial_values[var].set(1.0)
        self.error_tolerance.set(0.0001)
        self.max_iterations.set(50)
        self.tree.delete(*self.tree.get_children())
        self.message_var.set("")
        self.current_file = None
    
    def open_file(self):
        """Abre un archivo JSON con configuración guardada"""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not filepath:
             return
             
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Cargar ecuaciones
                for i, eq in enumerate(data.get('equations', [])):
                    self.equations[i].set(eq)
                
                # Cargar valores iniciales
                for var in self.vars:
                    self.initial_values[var].set(data['initial_values'].get(var, 1.0))
                
                # Cargar parámetros
                self.error_tolerance.set(data.get('tolerance', 0.0001))
                self.max_iterations.set(data.get('max_iterations', 50))
                
                self.current_file = filepath
                self.message_var.set(f"Archivo cargado: {os.path.basename(filepath)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def save_file(self):
        """Guarda la configuración actual en el archivo actual o abre diálogo si no hay uno"""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_as_file()
    
    def save_as_file(self):
        """Guarda la configuración actual en un nuevo archivo"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")]
        )
        if not filepath:
            return
            
        self._save_to_file(filepath)
        self.current_file = filepath
    
    def _save_to_file(self, filepath):
        """Guarda los datos actuales en el archivo especificado"""
        try:
            data = {
                'equations': [eq.get() for eq in self.equations],
                'initial_values': {var: val.get() for var, val in self.initial_values.items()},
                'tolerance': self.error_tolerance.get(),
                'max_iterations': self.max_iterations.get()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.message_var.set(f"Archivo guardado: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo:\n{str(e)}")
    

    def setup_ui(self):
        """Configura todos los componentes de la interfaz"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel de ecuaciones
        eq_frame = ttk.LabelFrame(main_frame, text="Sistema de 5 Ecuaciones", padding=10)
        eq_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        for i in range(5):
            ttk.Label(eq_frame, text=f"f{i+1}(x,y,z,w,r) =").grid(row=i, column=0, sticky='e', pady=2)
            ttk.Entry(eq_frame, textvariable=self.equations[i], width=28).grid(row=i, column=1, pady=2)
        
        # Panel de control
        control_frame = ttk.LabelFrame(main_frame, text="Parámetros", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Valores iniciales
        ttk.Label(control_frame, text="Valores Iniciales:").grid(row=0, column=0, columnspan=2, pady=5)
        for i, var in enumerate(self.vars):
            ttk.Label(control_frame, text=f"{var}₀ =").grid(row=i+1, column=0, sticky='e')
            ttk.Entry(control_frame, textvariable=self.initial_values[var], width=12).grid(row=i+1, column=1)
        
        # Configuración del método
        ttk.Label(control_frame, text="Tolerancia:").grid(row=6, column=0, sticky='e', pady=(15,2))
        ttk.Entry(control_frame, textvariable=self.error_tolerance, width=12).grid(row=6, column=1, pady=(15,2))
        
        ttk.Label(control_frame, text="Máx. Iteraciones:").grid(row=7, column=0, sticky='e')
        ttk.Entry(control_frame, textvariable=self.max_iterations, width=12).grid(row=7, column=1)
        
        # Botones de acción
        ttk.Button(control_frame, text="Resolver Sistema", 
                 command=self.solve_system,
                 style='Accent.TButton').grid(row=9, column=0, columnspan=2, pady=5)
         
        
        # Panel de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados por Iteración", padding=10)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Tabla de resultados
        columns = [('iter', 'Iteración', 70)] + [(var, var, 90) for var in self.vars] + [('error', 'Error', 90)]
        self.tree = ttk.Treeview(results_frame, columns=[col[0] for col in columns], show='headings')
        
        # Tabla de resultados con errores individuales
        columns = [
        ('iter', 'Iteración', 70),
        ('x', 'x', 90), ('y', 'y', 90),
        ('z', 'z', 90), ('w', 'w', 90),
        ('error_x', 'Error X', 100),  # Aumentado ancho para decimales
        ('error_y', 'Error Y', 100),
        ('error_z', 'Error Z', 100),
        ('error_w', 'Error W', 100),
        ('error_r', 'Error R', 100),
        ('error_total', 'Error Total', 110)
    ]

        self.tree = ttk.Treeview(results_frame, columns=[col[0] for col in columns], show='headings', height=18)
        
        for col_id, heading, width in columns:
            self.tree.heading(col_id, text=heading)
            self.tree.column(col_id, width=width, anchor=tk.CENTER)
        
        # Scrollbar y empaquetado
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Área de mensajes
        self.message_var = tk.StringVar()
        ttk.Label(results_frame, textvariable=self.message_var, foreground='red').pack(side=tk.BOTTOM, fill=tk.X)
    
    def parse_functions(self, eqs):
        """Convierte ecuaciones en funciones evaluables y calcula el Jacobiano"""
        x, y, z, w, r = symbols('x y z w r')
        sympy_funcs = []
        
        for eq in eqs:
            expr = sympify(eq.replace('^', '**').replace('ln', 'log'), 
                          locals={'x':x, 'y':y, 'z':z, 'w':w, 'r':r,
                                 'sin': lambda x: sympify('sin(x)'),
                                 'exp': lambda x: sympify('exp(x)'),
                                 'sqrt': lambda x: sympify('sqrt(x)')})
            sympy_funcs.append(expr)
        
        # Funciones numéricas
        numeric_funcs = [lambdify((x,y,z,w,r), f, 'numpy') for f in sympy_funcs]
        
        # Jacobiano (matriz de derivadas parciales)
        jacobian = Matrix([[diff(f, var) for var in [x,y,z,w,r]] for f in sympy_funcs])
        numeric_jacobian = lambdify((x,y,z,w,r), jacobian, 'numpy')
        
        return {
            'numeric_funcs': numeric_funcs,
            'numeric_jacobian': numeric_jacobian,
            'sympy_funcs': sympy_funcs,
            'jacobian': jacobian
        }
    
    def solve_system(self):
        """Ejecuta el método de Newton-Raphson para el sistema"""
        try:
            # Validar entrada
            eqs = [eq.get().strip() for eq in self.equations]
            if not all(eqs):
                raise ValueError("Debe ingresar las 5 ecuaciones")
            
            # Parsear funciones
            func_dict = self.parse_functions(eqs)
            
            # Obtener parámetros
            x0 = np.array([self.initial_values[var].get() for var in self.vars], dtype=float)
            tol = self.error_tolerance.get()
            max_iter = self.max_iterations.get()
            
            # Limpiar resultados previos
            self.tree.delete(*self.tree.get_children())
            self.message_var.set("")
            
            # Algoritmo de Newton-Raphson
            current = x0.copy()
            converged = False
            
            for iteration in range(max_iter):
                # Evaluar funciones y Jacobiano
                f_vals = np.array([f(*current) for f in func_dict['numeric_funcs']])
                J = func_dict['numeric_jacobian'](*current)
                
                # Verificar Jacobiano
                if np.abs(np.linalg.det(J)) < 1e-12:
                    raise np.linalg.LinAlgError("Jacobiano singular (det≈0)")
                
                # Paso de Newton
                try:
                    delta = np.linalg.solve(J, -f_vals)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError("No se puede resolver el sistema lineal")
                
                # Actualizar solución
                new = current + delta
                errors = np.abs(delta / (new + 1e-15))  # Evitar división por cero
                error_total = np.linalg.norm(delta) / np.linalg.norm(new)

                # Mostrar iteración
                values = [iteration+1] + [f"{val:.6f}" for val in current] 
                values += [f"{err:.6f}" for err in errors] + [f"{error_total:.6f}"]  # Cambiado a formato decimal
                self.tree.insert("", "end", values=values)
                
                # Verificar convergencia
                if all(err < tol for err in errors):
                    converged = True
                    break
                    
                current = new
            
            # Resultado final
            if converged:
               self.message_var.set(f"¡Convergencia alcanzada en {iteration+1} iteraciones!")
            else:
                self.message_var.set(f"Advertencia: Máximo de iteraciones alcanzado (Error total={error_total:.2e})")
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo resolver el sistema:\n{str(e)}")
            self.message_var.set(f"Error: {str(e)}")

# Ejecución principal
if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonRaphsonSystemSolver(root)
    root.mainloop()
