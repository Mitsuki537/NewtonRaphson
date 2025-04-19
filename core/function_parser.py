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
        self.x = symbols('x')
    
    def parse_function(self, func_str):
        """Analiza la función y devuelve un diccionario completo"""
        try:
            # Conservar el string original
            raw_str = func_str
            
            # Preparar para parsing
            parse_str = func_str.replace('^', '**').replace('√', 'sqrt')
            parse_str = parse_str.replace('log(', 'sympy_log10(').replace('ln(', 'sympy_ln(')
            
            expr = sympify(parse_str, locals={
                'sympy_ln': sympy_ln,
                'sympy_log10': lambda x: sympy_log(x, 10)
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
            
            derivative = diff(expr, self.x)
            
            return {
                'sympy_func': expr,
                'sympy_derivative': derivative,
                'numeric_func': lambdify(self.x, expr, modules=['numpy', modules]),
                'numeric_deriv': lambdify(self.x, derivative, modules=['numpy', modules]),
                'raw': raw_str,
                'pretty': self._make_pretty(raw_str)
            }
            
        except Exception as e:
            raise ValueError(f"Error al parsear función: {str(e)}")

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