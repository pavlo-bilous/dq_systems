from dataclasses import dataclass

import dynamiqs as dq

from .simple_system_abc import *
from .param_class import *


@dataclass(kw_only=True)
class ResonatorMode(SimpleSystemABC):
    omega: float | Callable
    omega_rwa: float | Callable
    
    
    def __post_init__(self):
        self.omega = Param(self.omega)
        self.omega_rwa = Param(self.omega_rwa)
        
    
    def Hamiltonian(self):
        H = self.omega * dq.number(self.N)
        H_rwa = self.omega_rwa * dq.number(self.N)
        return H - H_rwa