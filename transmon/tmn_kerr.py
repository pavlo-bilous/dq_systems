import dynamiqs as dq

from .tmn_abc import *
from ..fundamental import *
    

class TransmonKerr(TransmonABC):
    
    
    @staticmethod
    def eval_Ej(Ec, omega_qub):
        return (omega_qub + Ec)**2 / (8 * Ec)
        
      
    @staticmethod
    def eval_omega_qub(Ec, Ej):
        return jnp.sqrt(8 * Ec * Ej) - Ec
    
    
    def __init__(self, *,
                 N: int,
                 Ec: float,
                 omega_qub: float | Callable):
        
        self.linear_part = ResonatorMode(
            N=N,
            omega=omega_qub
        )
        
        if isinstance(omega_qub, Callable):
            Ej = lambda t: eval_Ej(Ec, omega_qub(t))
        else:
            Ej = TransmonKerr.eval_Ej(Ec, omega_qub)
        
        super().__init__(N, Ec, Ej)
    
    
    @property
    def K(self):
        return -self.Ec
    
    
    def Hamiltonian(self):
        H_lin = self.linear_part.Hamiltonian()
        H_nonlin = self.K / 2 * (
            dq.create(self.N) @ dq.create(self.N) @ dq.destroy(self.N) @ dq.destroy(self.N)
        )
        return H_lin + H_nonlin