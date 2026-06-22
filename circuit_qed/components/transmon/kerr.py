import jax.numpy as jnp
import dynamiqs as dq

from .base import *
    
    

class TransmonKerr(Transmon):
    
    def __init__(self, *, N: int, Ec: float, omega: float):
        Ej = (omega + Ec)**2 / (8 * Ec)
        super().__init__(N=N, Ec=Ec, Ej=Ej)
     
    
    @property
    def omega(self):
        return jnp.sqrt(8 * self.Ec * self.Ej) - self.Ec
    
    
    @property
    def K(self):
        return -self.Ec
    
    
    def Hamiltonian(self):
        H0 = self.omega * dq.number(self.N)
        H_nonlin = self.K / 2 * (
            dq.create(self.N) @ dq.create(self.N) @ dq.destroy(self.N) @ dq.destroy(self.N)
        )
        return H0 + H_nonlin