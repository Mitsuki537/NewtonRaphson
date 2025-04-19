from .result import NewtonRaphsonResult
from typing import List
import numpy as np 

class NewtonRaphsonSolver:
    def __init__(self, parser):
        self.parser = parser
        self.results = []

    def solve(self, func_dict: dict, x0: float, y0: float = None, 
              tol: float = 1e-6, max_iter: int = 50, 
              with_respect_to: str = 'x') -> List[NewtonRaphsonResult]:
        """
        Ejecuta el método Newton-Raphson para funciones de x o x e y
        
        Args:
            func_dict: Diccionario con la función y derivadas (del FunctionParser)
            x0: Valor inicial para x
            y0: Valor inicial para y (opcional)
            tol: Tolerancia de error
            max_iter: Máximo de iteraciones
            with_respect_to: Variable respecto a la cual derivar ('x' o 'y')
            
        Returns:
            Lista de objetos NewtonRaphsonResult con los resultados por iteración
        """
        self.results = []
        x = x0
        y = y0 if y0 is not None else 0  # Valor por defecto para y si no se especifica
        iter_count = 0
        
        while iter_count < max_iter:
            try:
                # Evaluar función y derivada adecuada
                if with_respect_to == 'x':
                    fx = self.parser.evaluate(func_dict, x, y)
                    dfx = self.parser.evaluate_derivative_x(func_dict, x, y)
                    x_new = x - fx/dfx if dfx != 0 else x
                    error = abs((x_new - x)/x_new) if x_new != 0 else float('inf')
                    y_new = y  # y permanece constante
                elif with_respect_to == 'y' and 'y' in func_dict['variables']:
                    fx = self.parser.evaluate(func_dict, x, y)
                    dfy = self.parser.evaluate_derivative_y(func_dict, x, y)
                    y_new = y - fx/dfy if dfy != 0 else y
                    error = abs((y_new - y)/y_new) if y_new != 0 else float('inf')
                    x_new = x  # x permanece constante
                else:
                    raise ValueError("Variable no soportada para derivación")
                
                # Guardar resultado
                result = NewtonRaphsonResult(
                    iteration=iter_count+1,
                    x_prev=x,
                    y_prev=y if 'y' in func_dict['variables'] else None,
                    fx=fx,
                    dfx=dfx if with_respect_to == 'x' else None,
                    dfy=dfy if with_respect_to == 'y' else None,
                    x_new=x_new,
                    y_new=y_new if 'y' in func_dict['variables'] else None,
                    error=error
                )
                self.results.append(result)
                
                # Verificar convergencia
                if error < tol:
                    break
                    
                x, y = x_new, y_new
                iter_count += 1
                
            except ValueError as e:
                break
                
        return self.results

    # Nuevos métodos para evaluación específica
    def evaluate(self, func_dict, x_val, y_val=0):
        """Evalúa la función en (x_val, y_val)"""
        return float(func_dict['numeric_func'](x_val, y_val))
    
    def evaluate_derivative_x(self, func_dict, x_val, y_val=0):
        """Evalúa la derivada respecto a x en (x_val, y_val)"""
        return float(func_dict['numeric_deriv_x'](x_val, y_val))
    
    def evaluate_derivative_y(self, func_dict, x_val, y_val=0):
        """Evalúa la derivada respecto a y en (x_val, y_val)"""
        if func_dict['numeric_deriv_y'] is None:
            raise ValueError("La función no contiene la variable y")
        return float(func_dict['numeric_deriv_y'](x_val, y_val))

    def get_last_result(self) -> NewtonRaphsonResult:
        """Devuelve el último resultado calculado"""
        return self.results[-1] if self.results else None