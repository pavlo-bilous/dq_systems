import dynamiqs as dq

from .abc import *
from ..fundamental import *
    

class TransmonKerr(Transmon):
    
    def __init__(self, *,
                 N: int,
                 Ec: float,
                 omega_qub: float,
                 omega_rwa: float = 0.0):
        
        self.linear_part = ResonatorMode(
            N=N,
            omega=omega_qub,
            omega_rwa=omega_rwa
        )
        
        Ej = (omega_qub + Ec)**2 / (8 * Ec)
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