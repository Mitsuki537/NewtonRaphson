from dataclasses import dataclass

@dataclass
class NewtonRaphsonResult:
    iteration: int
    x_prev: float
    fx: float
    dfx: float
    x_new: float
    error: float
    
    def to_dict(self):
        """Convierte el resultado a diccionario para la GUI"""
        return {
            'iteration': self.iteration,
            'x_prev': self.x_prev,
            'fx': self.fx,
            'dfx': self.dfx,
            'x_new': self.x_new,
            'error': self.error
        }