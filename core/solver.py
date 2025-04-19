from .result import NewtonRaphsonResult
from typing import List
import numpy as np

class NewtonRaphsonSolver:
    def __init__(self, parser):
        self.parser = parser
        self.results = []

    def solve(self, func_dict: dict, x0: float, tol: float = 1e-6, max_iter: int = 50) -> List[NewtonRaphsonResult]:
        """
        Ejecuta el método Newton-Raphson
        
        Args:
            func_dict: Diccionario con la función y derivada (del FunctionParser)
            x0: Valor inicial
            tol: Tolerancia de error
            max_iter: Máximo de iteraciones
            
        Returns:
            Lista de objetos NewtonRaphsonResult con los resultados por iteración
        """
        self.results = []
        x = x0
        iter_count = 0
        
        while iter_count < max_iter:
            # Evaluar función y derivada
            try:
                fx = self.parser.evaluate(func_dict, x)
                dfx = self.parser.evaluate_derivative(func_dict, x)
            except ValueError:
                break

            # Calcular nuevo x y error
            x_new = x - fx/dfx if dfx != 0 else x
            error = abs((x_new - x)/x_new) if x_new != 0 else float('inf')
            
            # Guardar resultado
            result = NewtonRaphsonResult(
                iteration=iter_count+1,
                x_prev=x,
                fx=fx,
                dfx=dfx,
                x_new=x_new,
                error=error
            )
            self.results.append(result)
            
            # Verificar convergencia
            if error < tol:
                break
                
            x = x_new
            iter_count += 1
            
        return self.results

    def get_last_result(self) -> NewtonRaphsonResult:
        """Devuelve el último resultado calculado"""
        return self.results[-1] if self.results else None