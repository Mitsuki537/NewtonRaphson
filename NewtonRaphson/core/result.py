from dataclasses import dataclass
from typing import Optional

@dataclass
class NewtonRaphsonResult:
    """
    Almacena los resultados de cada iteración del método Newton-Raphson,
    ahora soportando funciones multivariables (x e y).
    """
    iteration: int
    x_prev: float
    fx: float
    x_new: float
    error: float
    
    # Campos opcionales para funciones con y
    y_prev: Optional[float] = None
    dfx: Optional[float] = None      # Derivada respecto a x
    dfy: Optional[float] = None      # Derivada respecto a y
    y_new: Optional[float] = None
    
    def __repr__(self):
        base = (f"Iteration {self.iteration}:\n"
                f"  x_prev = {self.x_prev:.6f}\n"
                f"  y_prev = {self.y_prev:.6f}\n" if self.y_prev is not None else ""
                f"  f(x,y) = {self.fx:.6f}\n")
        
        if self.dfy is not None:
            base += f"  df/dy = {self.dfy:.6f}\n"
        if self.dfx is not None:
            base += f"  df/dx = {self.dfx:.6f}\n"
            
        base += (f"  x_new = {self.x_new:.6f}\n" +
                (f"  y_new = {self.y_new:.6f}\n" if self.y_new is not None else "") +
                f"  error = {self.error:.6%}")
        
        return base

    def to_dict(self):
        """Convierte el resultado a diccionario para fácil serialización"""
        return {
            'iteration': self.iteration,
            'x_prev': self.x_prev,
            'y_prev': self.y_prev,
            'fx': self.fx,
            'dfx': self.dfx,
            'dfy': self.dfy,
            'x_new': self.x_new,
            'y_new': self.y_new,
            'error': self.error
        }