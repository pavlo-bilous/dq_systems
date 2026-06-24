from dataclasses import dataclass

import dynamiqs as dq

from .simple_system_abc import *
from .param_class import *


@dataclass(kw_only=True)
class ResonatorMode(SimpleSystemABC):
    omega: float | Callable
    
    
    def __post_init__(self):
        self.omega = Param(self.omega)
        
    
    def Hamiltonian(self):
        return self.omega * dq.number(self.N)