import sys
from pathlib import Path

# Configuración esencial de rutas
sys.path.append(str(Path(__file__).parent.parent))

# Ahora los imports normales
from core.function_parser import FunctionParser
from core.solver import NewtonRaphsonSolver
from core.result import NewtonRaphsonResult
from utils.plotter import Plotter

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, sympify, diff, pi, factorial

class NewtonRaphson:
    def __init__(self, root):
        self.root = root
        self.root.title("Newton-Raphson")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')
        
        # Paleta de colores 
        self.colors = {
            'primary': '#4e73df',
            'secondary': '#858796',
            'success': '#1cc88a',
            'info': '#36b9cc',
            'warning': '#f6c23e',
            'danger': '#e74a3b',
            'light': '#f8f9fc',
            'dark': '#5a5c69',
            'bg': '#f8f9fa',
            'panel': '#ffffff',
            'text': '#3a3b45',
            'border': '#d1d3e2'
        }
        

        self.parser = FunctionParser()  
        self.solver = NewtonRaphsonSolver(self.parser)
        self.plotter = Plotter()
        # Variables
        self.functions = []
        self.current_function = None
        self.error_tolerance = tk.DoubleVar(value=0.01)
        self.initial_value = tk.DoubleVar(value=1.0)
        
        # Configurar estilos 
        self.setup_styles()
        
        # Crear widgets con diseño 
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo general
        style.configure('.', 
                      background=self.colors['bg'],
                      foreground=self.colors['text'],
                      font=('Segoe UI', 10))
        
        # Estilo para botones
        style.configure('Modern.TButton',
                      background=self.colors['primary'],
                      foreground='white',
                      font=('Segoe UI', 10),
                      borderwidth=0,
                      focusthickness=0,
                      padding=10,
                      relief='flat',
                      anchor='center',
                      bordercolor=self.colors['primary'],
                      lightcolor=self.colors['primary'],
                      darkcolor=self.colors['primary'])
        
        style.map('Modern.TButton',
                background=[('active', self.colors['info']),
                          ('pressed', self.colors['success']),
                          ('!disabled', self.colors['primary'])],
                foreground=[('disabled', '#cccccc'),
                          ('!disabled', 'white')])
        
        # Estilo para entrada de función 
        style.configure('Modern.TEntry',
                      fieldbackground=self.colors['panel'],
                      foreground=self.colors['text'],
                      relief='flat',
                      borderwidth=1,
                      bordercolor='#d1d3e2',
                      focusthickness=1,
                      focuscolor=self.colors['info'],
                      padding=10,
                      font=('Consolas', 11))
        
        # Estilo para el notebook (pestañas)
        style.configure('Modern.TNotebook',
                      background=self.colors['bg'],
                      borderwidth=0)
        
        style.configure('Modern.TNotebook.Tab',
                      background=self.colors['light'],
                      foreground=self.colors['secondary'],
                      padding=(15, 5),
                      font=('Segoe UI', 9),
                      borderwidth=0,
                      lightcolor=self.colors['light'],
                      darkcolor=self.colors['light'])
        
        style.map('Modern.TNotebook.Tab',
                background=[('selected', self.colors['panel']),
                          ('!selected', self.colors['light'])],
                foreground=[('selected', self.colors['primary']),
                          ('!selected', self.colors['secondary'])])
        
        # Estilo para el Treeview (tabla de resultados)
        style.configure('Modern.Treeview',
                      background=self.colors['panel'],
                      foreground=self.colors['text'],
                      fieldbackground=self.colors['panel'],
                      bordercolor='#dddfeb',
                      borderwidth=0,
                      rowheight=30,
                      font=('Segoe UI', 9))
        
        style.configure('Modern.Treeview.Heading',
                      background=self.colors['primary'],
                      foreground='white',
                      font=('Segoe UI', 9),
                      padding=5,
                      relief='flat')
        
        style.map('Modern.Treeview',
                 background=[('selected', '#e8f4ff')],
                 foreground=[('selected', self.colors['dark'])])


         # Configuración de colores para scrollbar
        bg_color = '#f0f0f0' 
        thumb_color = '#909090'  
        scroll_width = 8  

            # Resetear estilos previos completamente
        style.layout('Vertical.TScrollbar', [])
        style.layout('Horizontal.TScrollbar', [])
    
            
        # Scrollbar Vertical
        style.element_create('Vertical.Scrollbar.trough', 'from', 'clam')
        style.element_create('Vertical.Scrollbar.thumb', 'from', 'clam')
        
        style.layout('Vertical.TScrollbar',
            [('Vertical.Scrollbar.trough', {
                'sticky': 'ns',
                'children': [('Vertical.Scrollbar.thumb', {
                    'sticky': 'ns',
                    'unit': '1'
                })]
            })]
        )
        
        style.configure('Vertical.TScrollbar',
            background=thumb_color,
            troughcolor=bg_color,
            bordercolor=bg_color,  
            relief='flat',        
            width=scroll_width
        )
        
        # Scrollbar Horizontal
        style.element_create('Horizontal.Scrollbar.trough', 'from', 'clam')
        style.element_create('Horizontal.Scrollbar.thumb', 'from', 'clam')
        
        style.layout('Horizontal.TScrollbar',
            [('Horizontal.Scrollbar.trough', {
                'sticky': 'ew',
                'children': [('Horizontal.Scrollbar.thumb', {
                    'sticky': 'ew',
                    'unit': '1'
                })]
            })]
        )
        
        style.configure('Horizontal.TScrollbar',
            background=thumb_color,
            troughcolor=bg_color,
            bordercolor=bg_color,
            relief='flat',
            height=scroll_width,
            arrowsize=0
        )


    def create_widgets(self):
        # Frame principal con diseño 
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Panel izquierdo - Controles
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Panel derecho - Visualización
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # --- Panel de entrada de función ---
        input_frame = ttk.Frame(left_panel, padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Título
        ttk.Label(input_frame, text="Ingrese una función", 
                 font=('Segoe UI', 10, 'bold'), 
                 foreground=self.colors['primary']).pack(anchor='w', pady=(0, 10))
        
        # Entrada de función 
        self.func_entry = ttk.Entry(input_frame, style='Modern.TEntry')
        self.func_entry.pack(fill=tk.X, pady=(0, 10))
        self.func_entry.bind('<KeyRelease>', self.update_function_preview)
        
        # Vista previa matemática 
        self.func_preview = ttk.Label(input_frame, text="f(x) = ", 
                                     font=('Cambria Math', 14, 'bold'), 
                                     foreground=self.colors['primary'],
                                     background=self.colors['panel'])
        self.func_preview.pack(fill=tk.X, pady=(0, 15))
        
        # Frame para parámetros
        params_frame = ttk.Frame(input_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Valor inicial
        ttk.Label(params_frame, text="Valor inicial:").pack(side=tk.LEFT)
        self.initial_entry = ttk.Entry(params_frame, textvariable=self.initial_value, width=8)
        self.initial_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        # Error tolerance
        ttk.Label(params_frame, text="Error (%):").pack(side=tk.LEFT)
        self.error_entry = ttk.Entry(params_frame, textvariable=self.error_tolerance, width=8)
        self.error_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Botón para agregar función
        self.add_btn = ttk.Button(input_frame, text="Agregar función",
                                style='Modern.TButton',
                                command=self.add_function)
        self.add_btn.pack(fill=tk.X, pady=(5, 0))
        
        # --- Teclado matemático ---
        self.create_math_keyboard(left_panel)
        
        # --- Lista de funciones simplificada ---
        self.create_functions_list(left_panel)
        
        # --- Panel de gráfico ---
        self.create_modern_graph(right_panel)
        
        # --- Panel de resultados ---
        self.create_results_table(right_panel)

    def create_math_keyboard(self, parent):
        """Teclado matemático"""
        keyboard_frame = ttk.Frame(parent, style='Modern.TFrame', padding=15)
        keyboard_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Notebook con pestañas 
        notebook = ttk.Notebook(keyboard_frame, style='Modern.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de números y operaciones básicas
        num_tab = ttk.Frame(notebook, style='Modern.TFrame')
        notebook.add(num_tab, text="Números")
        
        # Diseño de teclado 
        num_buttons = [
            ['7', '8', '9', '+', '(', ')'],
            ['4', '5', '6', '-', '^', '√'],
            ['1', '2', '3', '*', 'π', 'e'],
            ['0', '.', '=', '/', 'x', 'y']
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
                btn_widget.bind('<Enter>', lambda e, b=btn_widget: 
                               b.config(bg='#e8f4ff'))
                btn_widget.bind('<Leave>', lambda e, b=btn_widget: 
                               b.config(bg=self.colors['panel']))
        
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
                btn_widget.bind('<Enter>', lambda e, b=btn_widget: 
                              b.config(bg='#e8f4ff'))
                btn_widget.bind('<Leave>', lambda e, b=btn_widget: 
                              b.config(bg=self.colors['panel']))

    def create_functions_list(self, parent):
        """Lista de funciones con scroll que mantiene posición al agregar items"""
        container = ttk.Frame(parent, style='Modern.TFrame', padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # Título
        ttk.Label(container, text="Funciones Guardadas", 
                font=('Segoe UI', 10, 'bold'), 
                foreground=self.colors['primary']).pack(anchor='w', pady=(0, 10))

        # Frame contenedor principal
        list_container = ttk.Frame(container)
        list_container.pack(fill=tk.BOTH, expand=True)

        # Canvas para scroll
        canvas = tk.Canvas(
            list_container,
            bg=self.colors['panel'],
            highlightthickness=0,
            bd=0
        )
        
        # Scrollbar 
        scrollbar = ttk.Scrollbar(
            list_container,
            orient="vertical",
            command=canvas.yview,
            style='Custom.Vertical.TScrollbar'
        )

        # Frame scrollable
        scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')
        
        # Configurar canvas y scroll
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=self.update_scrollbar)
        
        def configure_scroll(event):
            """Ajusta el scrollregion cuando cambia el tamaño del frame"""
            canvas.itemconfig(1, width=canvas.winfo_width())
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
        
        scrollable_frame.bind("<Configure>", configure_scroll)
        
        # Empaquetado
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configurar scroll con rueda del mouse
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # Prevenir propagación del evento
        
        canvas.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)

        # Guardar referencias importantes
        self.scrollable_functions_frame = scrollable_frame
        self.functions_canvas = canvas
        self.functions_scrollbar = scrollbar


    def create_modern_graph(self, parent):
        """Gráfico"""
        graph_frame = ttk.Frame(parent, style='Modern.TFrame', padding=10)
        graph_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(graph_frame, text="Gráfica de la función", 
                 font=('Segoe UI', 10, 'bold'), 
                 foreground=self.colors['primary']).pack(anchor='w', pady=(0, 10))
        
        # Crear figura de matplotlib 
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('ggplot')
            
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.fig.patch.set_facecolor(self.colors['panel'])
        self.ax.set_facecolor(self.colors['panel'])
        
        # Configuración del gráfico
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_color(self.colors['secondary'])
        self.ax.spines['left'].set_color(self.colors['secondary'])
        
        # Canvas para el gráfico
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_scrollbar(self, *args):
        """Controla el comportamiento del scrollbar al agregar funciones"""
        # Actualizar posición del scrollbar
        self.functions_scrollbar.set(*args)
        
        if float(args[1]) < 0.9: 
            current_pos = self.functions_canvas.yview()[0]
            self.functions_canvas.yview_moveto(current_pos)

    def create_results_table(self, parent):
        """Tabla de resultados"""
        results_frame = ttk.Frame(parent, style='Modern.TFrame', padding=10)
        results_frame.pack(fill=tk.BOTH, pady=(15, 0))
        
        # Título
        ttk.Label(results_frame, text="Resultados del método", 
                 font=('Segoe UI', 10, 'bold'), 
                 foreground=self.colors['primary']).pack(anchor='w', pady=(0, 10))
        
        # Crear Treeview con estilo 
        self.tree = ttk.Treeview(results_frame, 
                                columns=("i", "Xi-1", "f(Xi-1)", "f'(Xi-1)", "Xi", "Error%"), 
                                show="headings", 
                                height=6,
                                style='Modern.Treeview')
        
        # Configurar columnas
        cols = [
            ("i", "Iteración", 70),
            ("Xi-1", "Xi-1", 100),
            ("f(Xi-1)", "f(Xi-1)", 120),
            ("f'(Xi-1)", "f'(Xi-1)", 120),
            ("Xi", "Xi", 100),
            ("Error%", "Error %", 80)
        ]
        
        for col_id, heading, width in cols:
            self.tree.heading(col_id, text=heading)
            self.tree.column(col_id, width=width, anchor=tk.CENTER)
        

        scrollbar = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.tree.yview,
            style='Vertical.TScrollbar'
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)

        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Empaquetado
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)



    def insert_math_symbol(self, symbol):
        """Inserta símbolos matemáticos en la entrada"""
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
            'log': 'log(',
            'ln': 'ln(',
            'exp': 'exp('
        }
        # Obtener texto antes y después del cursor
        cursor_pos = self.func_entry.index(tk.INSERT)
        current_text = self.func_entry.get()
        before = current_text[:cursor_pos]
        after = current_text[cursor_pos:]
        
        # Insertar el símbolo mapeado
        text_to_insert = mapping.get(symbol, symbol)
        
        # Caso especial: si insertamos función después de letra, agregar *
        if text_to_insert.endswith('(') and before and before[-1].isalpha():
            text_to_insert = '*' + text_to_insert
        
        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, before + text_to_insert + after)
        
        # Posicionar cursor
        if text_to_insert.endswith('('):
            new_pos = cursor_pos + len(text_to_insert)
            self.func_entry.icursor(new_pos)
        
        self.update_function_preview()



    def update_function_preview(self, event=None):
        """Actualiza la vista previa matemática"""
        func_text = self.func_entry.get()
        
        try:
            # Procesar la función (acepta f(x)=x o g(x)=x)
            if '=' in func_text:
                func_name, expr_part = func_text.split('=', 1)
                func_name = func_name.strip()
                expr_part = expr_part.strip()
            else:
                func_name = 'f(x)'
                expr_part = func_text
                
            x = symbols('x')
            expr = sympify(expr_part)
            
            # Crear representación 
            pretty_text = f"{func_name} = {str(expr).replace('**', '^')}"
            pretty_text = pretty_text.replace('sqrt(', '√(').replace('pi', 'π')
            
            self.func_preview.config(text=pretty_text)
        except:
            # Si hay error, mostrar el texto crudo
            self.func_preview.config(text=f"f(x) = {func_text}")

    
    def add_function(self): func_text = self.func_entry.get().strip()
    
    if not func_text:
        messagebox.showerror("Error", "No se ingresó ninguna función")  
    
    try:
        # Parsear y crear estructura completa de función
        func_dict = self.parser.parse_function(func_text)
        
        # Verificar explícitamente que la derivada se calculó
        if 'sympy_derivative' not in func_dict:
            raise ValueError("No se pudo calcular la derivada de la función")
            
        new_func = {
            'raw': func_text,
            'data': func_dict,
            'initial_value': self.initial_value.get(),
            'tolerance': self.error_tolerance.get(),
            'pretty': self.parser.get_human_readable(func_dict),  
            'derivative': str(func_dict['sympy_derivative']) 
        }
        
        self.functions.append(new_func)
        self.update_functions_list()
        self.func_entry.delete(0, tk.END) 
        
    except Exception as e:
        messagebox.showerror("Error", f"Error al agregar función:\n{str(e)}")
        # Imprimir el error en consola para depuración
        print(f"Error detallado: {traceback.format_exc()}")


    def update_functions_list(self):
        """Actualiza la lista visual de funciones con selección correcta"""
        # Limpiar frame existente
        for widget in self.scrollable_functions_frame.winfo_children():
            widget.destroy()
        
        if not self.functions:
            # Mostrar mensaje si no hay funciones
            empty_label = ttk.Label(
                self.scrollable_functions_frame,
                text="No hay funciones guardadas",
                foreground=self.colors['secondary'],
                font=('Segoe UI', 9, 'italic')
            )
            empty_label.pack(pady=10)
            return
        
        # Crear un frame contenedor para mantener el ancho consistente
        container_width = self.functions_canvas.winfo_width() - 20
        
        for i, func in enumerate(self.functions):
            # Frame para cada función
            func_frame = tk.Frame(
                self.scrollable_functions_frame,
                bg=self.colors['panel'],
                highlightbackground=self.colors['border'],
                highlightthickness=1,
                bd=0
            )
            func_frame.pack(fill=tk.X, pady=(0, 5), padx=(0, 5))
            
            # Punto de selección - azul solo para la función seleccionada actualmente
            is_selected = (i == self.current_function)
            dot_color = self.colors['primary'] if is_selected else self.colors['light']
            
            select_dot = tk.Canvas(
                func_frame,
                width=20,
                height=20,
                bg=self.colors['panel'],
                highlightthickness=0,
                bd=0
            )
            select_dot.create_oval(2, 2, 18, 18, fill=dot_color, outline=self.colors['border'])
            select_dot.pack(side=tk.LEFT, padx=(5, 10))
            select_dot.bind('<Button-1>', lambda e, idx=i: self.select_function(idx))
            
            # Label con la función
            func_label = tk.Label(
                func_frame,
                text=func['pretty'],
                bg=self.colors['panel'],
                fg=self.colors['text'],
                font=('Segoe UI', 10),
                anchor='w',
                width=container_width // 8,
                wraplength=container_width - 50
            )
            func_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            func_label.bind('<Button-1>', lambda e, idx=i: self.select_function(idx))
            
            # Botón de eliminar
            trash_btn = tk.Label(
                func_frame,
                text="✕",
                bg=self.colors['panel'],
                fg=self.colors['secondary'],
                font=('Segoe UI', 14),
                cursor="hand2"
            )
            trash_btn.pack(side=tk.RIGHT, padx=(0, 5))
            trash_btn.bind('<Button-1>', lambda e, idx=i: self.delete_function(idx))
            
            # Efectos hover
            for widget in [func_frame, func_label, select_dot, trash_btn]:
                widget.bind('<Enter>', lambda e, f=func_frame: f.config(bg='#f0f2f5'))
                widget.bind('<Leave>', lambda e, f=func_frame: f.config(bg=self.colors['panel']))
        
        # Actualizar el scroll
        self.functions_canvas.configure(scrollregion=self.functions_canvas.bbox("all"))

    def select_function(self, index):
        """Selección mejorada con validación robusta"""
        if not 0 <= index < len(self.functions):
            return
            
        self.current_function = index
        func = self.functions[index]
        
        try:
            # Actualizar interfaz
            self.initial_value.set(func['initial_value'])
            self.error_tolerance.set(func['tolerance'])
            self.func_preview.config(text=func['pretty'])
            
            # Actualizar gráfico y tabla
            self.clear_and_plot()
            
            # Actualizar la lista visual para reflejar la nueva selección
            self.update_functions_list()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al seleccionar función:\n{str(e)}")
            

    def delete_function(self, index):
        """Elimina una función de la lista"""
        if 0 <= index < len(self.functions):
            self.functions.pop(index)
            self.update_functions_list()
            
            if self.current_function == index:
                self.current_function = None
                self.ax.clear()
                self.canvas_graph.draw()


    def clear_and_plot(self):
        """Limpia resultados previos y grafica la función actual"""
        # Limpiar tabla completamente
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Limpiar gráfico
        self.ax.clear()
        
        if self.current_function is not None:
            func = self.functions[self.current_function]
            
            try:
                # Calcular con los parámetros específicos de esta función
                results = self.solver.solve(
                    func_dict=func['data'],
                    x0=func['initial_value'],
                    tol=func['tolerance']/100,
                    max_iter=20
                )
                
                # Configurar rango dinámico para la gráfica
                x_values = [res.x_prev for res in results] + [res.x_new for res in results]
                x_min = min(x_values) - 1
                x_max = max(x_values) + 1
                x_vals = np.linspace(x_min, x_max, 400)
                
                # Graficar función principal
                y_vals = [func['data']['numeric_func'](x) for x in x_vals]
                self.ax.plot(x_vals, y_vals, label=func['pretty'], linewidth=2)
                
                # Graficar iteraciones del método
                for i, res in enumerate(results):
                    color = plt.cm.viridis(i/len(results))  # Color progresivo
                    self.ax.plot(res.x_prev, res.fx, 'o', color=color, markersize=8)
                    self.ax.plot([res.x_prev, res.x_new], [res.fx, 0], '--', color=color)
                    self.ax.plot(res.x_new, 0, 's', color=color, markersize=8)
                
                # Configuraciones finales del gráfico
                self.ax.axhline(0, color='black', linewidth=0.5)
                self.ax.axvline(0, color='black', linewidth=0.5)
                self.ax.grid(True, alpha=0.3)
                self.ax.legend()
                self.ax.set_title(f"Aproximación de raíz para {func['pretty']}")
                
                # Mostrar resultados en tabla
                self.display_results(results)
                
                # Redibujar el canvas
                self.canvas_graph.draw()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al calcular: {str(e)}")

        def update_parameters(self):
            """Actualiza los parámetros para la función actual"""
            if self.current_function is not None:
                # Actualizar valores guardados
                self.functions[self.current_function]['initial_value'] = self.initial_value.get()
                self.functions[self.current_function]['tolerance'] = self.error_tolerance.get()
                
                # Recalcular y graficar
                self.clear_and_plot()

    def plot_function(self):
        """Graficación con manejo de errores mejorado"""
        if self.current_function is None:
            return
            
        func_data = self.functions[self.current_function]
        
        try:
            self.ax.clear()
            results = self.solver.solve(
                func_dict=func_data['data'],
                x0=func_data['initial_value'],  # Usar valor guardado
                tol=func_data['tolerance']/100,
                max_iter=20
            )
            
            # Configurar rango dinámico
            x_values = [res.x_prev for res in results] + [res.x_new for res in results]
            x_min, x_max = min(x_values)-1, max(x_values)+1
            x_vals = np.linspace(x_min, x_max, 400)
            
            # Evaluar y graficar
            y_vals = [func_data['data']['numeric_func'](x) for x in x_vals]
            self.ax.plot(x_vals, y_vals, label=func_data['pretty'])
            
            # Graficar iteraciones
            for res in results:
                self.ax.plot(res.x_prev, res.fx, 'ro')
                self.ax.plot([res.x_prev, res.x_new], [res.fx, 0], 'r--')
                self.ax.plot(res.x_new, 0, 'go')
            
            self.ax.legend()
            self.display_results(results)
            self.canvas_graph.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar:\n{str(e)}")


    def calculate_newton_approximation(self):
        """Cálculo con validación mejorada"""
        if self.current_function is None:
            messagebox.showwarning("Advertencia", "No hay función seleccionada")
            return
            
        func_data = self.functions[self.current_function]
        
        try:
            # Limpiar resultados anteriores
            self.tree.delete(*self.tree.get_children())
            self.ax.clear()
            
            # Calcular
            results = self.solver.solve(
                func_dict=func_data['data'],
                x0=func_data['initial_value'],
                tol=func_data['tolerance']/100,
                max_iter=20
            )
            
            # Mostrar y graficar resultados
            self.display_results(results)
            self.plot_function() 
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo:\n{str(e)}")


    def display_results(self, results):
        """Muestra los resultados en la tabla Treeview con '-' en el primer error"""
        # Limpiar tabla primero
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Insertar nuevos resultados
        for i, res in enumerate(results):
            error_display = "-" if i == 0 else f"{res.error*100:.6f}%"
            self.tree.insert("", "end", values=(
                res.iteration,
            f"{res.x_prev:.6f}",
            f"{res.y_prev:.6f}" if res.y_prev is not None else "-",
            f"{res.fx:.6f}",
            f"{res.dfx:.6f}" if res.dfx is not None else "-",
            f"{res.dfy:.6f}" if res.dfy is not None else "-",
            f"{res.x_new:.6f}",
            f"{res.y_new:.6f}" if res.y_new is not None else "-",
            f"{res.error:.6%}" if res.iteration > 1 else "-",
                error_display
            ))    

if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonRaphson(root)
    root.mainloop()