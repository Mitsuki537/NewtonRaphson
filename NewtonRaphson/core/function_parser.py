from sympy import sympify, symbols, diff, lambdify, exp, log as sympy_log, ln as sympy_ln, sqrt
import numpy as np
from math import factorial as math_factorial
from sympy.parsing.sympy_parser import (
    standard_transformations,
    implicit_multiplication,
    convert_xor
)

class FunctionParser:
    def __init__(self):
        self.transformations = (
            standard_transformations +
            (implicit_multiplication, convert_xor)
        )
        self.x, self.y = symbols('x y')  # Ahora soporta x e y
    
    def parse_function(self, func_str):
        """Analiza la función y devuelve un diccionario completo, ahora soporta y"""
        try:
            raw_str = func_str
            
            # Preparar para parsing
            parse_str = func_str.replace('^', '**').replace('√', 'sqrt')
            parse_str = parse_str.replace('log(', 'sympy_log10(').replace('ln(', 'sympy_ln(')
            
            expr = sympify(parse_str, locals={
                'sympy_ln': sympy_ln,
                'sympy_log10': lambda x: sympy_log(x, 10),
                'y': self.y  # Añadimos y a los símbolos conocidos
            })
            
            # Configuración numérica
            modules = {
                'sqrt': np.sqrt,
                'exp': np.exp,
                'ln': np.log,
                'log': np.log10,
                'sympy_ln': np.log,
                'sympy_log10': np.log10,
                'factorial': math_factorial,
                'pi': np.pi,
                'abs': np.abs
            }
            
            # Calcular derivadas parciales
            derivative_x = diff(expr, self.x)
            derivative_y = diff(expr, self.y)
            
            # Determinar si es función multivariable
            variables = ['x']
            if self.y in expr.free_symbols:
                variables.append('y')
            
            return {
                'sympy_func': expr,
                'sympy_derivative_x': derivative_x,
                'sympy_derivative_y': derivative_y if 'y' in variables else None,
                'numeric_func': lambdify((self.x, self.y), expr, modules=['numpy', modules]),
                'numeric_deriv_x': lambdify((self.x, self.y), derivative_x, modules=['numpy', modules]),
                'numeric_deriv_y': lambdify((self.x, self.y), derivative_y, modules=['numpy', modules]) if 'y' in variables else None,
                'raw': raw_str,
                'pretty': self._make_pretty(raw_str),
                'variables': variables
            }
            
        except Exception as e:
            raise ValueError(f"Error al parsear función: {str(e)}")

    # ... (resto de métodos permanecen igual)

    def _make_pretty(self, func_str):
        """Crea versión legible sin cambiar la semántica"""
        return (func_str.replace('**', '^')
                      .replace('sqrt', '√')
                      .replace('sympy_ln', 'ln')
                      .replace('sympy_log10', 'log'))

    def get_human_readable(self, func_dict):
        """Versión segura para obtener representación legible"""
        return func_dict.get('pretty', str(func_dict.get('sympy_func', '')))

    def evaluate(self, func_dict, x_value):
        """Evalúa la función numéricamente"""
        try:
            return float(func_dict['numeric_func'](x_value))
        except Exception as e:
            raise ValueError(f"Error al evaluar: {str(e)}")

    def evaluate_derivative(self, func_dict, x_value):
        """Evalúa la derivada numéricamente"""
        try:
            return float(func_dict['numeric_deriv'](x_value))
        except Exception as e:
            raise ValueError(f"Error al evaluar derivada: {str(e)}")