"""
plotter.py - Generación de gráficos para el método
"""
import matplotlib.pyplot as plt
import numpy as np
class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.setup_plot_style()
        
    def setup_plot_style(self):
        """Configura el estilo del gráfico"""
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.fig.patch.set_facecolor('#f8f9fa')
        self.ax.set_facecolor('#ffffff')
        
    def plot_function(self, func_dict, x_vals, results=None):
        """Grafica la función y las iteraciones"""
        self.ax.clear()
        self.setup_plot_style()
        
        # Graficar función principal
        y_vals = [func_dict['numeric_func'](x) for x in x_vals]
        self.ax.plot(x_vals, y_vals, label='f(x)', linewidth=2)
        
        # Graficar iteraciones si existen
        if results:
            for i, res in enumerate(results):
                # Punto actual
                self.ax.plot(res.x_prev, res.fx, 'ro', label=f'Iter {i+1}' if i == 0 else "")
                # Línea de aproximación
                self.ax.plot([res.x_prev, res.x_new], [res.fx, 0], 'r--')
                # Nueva aproximación
                self.ax.plot(res.x_new, 0, 'go')
        
        self.ax.legend()
        
    def show(self):
        """Muestra el gráfico"""
        plt.show()