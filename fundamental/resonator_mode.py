from dataclasses import dataclass

import dynamiqs as dq

from .simple_system_abc import *


@dataclass(kw_only=True)
class ResonatorMode(SimpleSystemABC):
    omega: float
    omega_rwa: float = 0.0
    
    
    def Hamiltonian(self):
        return (self.omega - self.omega_rwa) * dq.number(self.N)