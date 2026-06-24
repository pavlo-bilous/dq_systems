from dataclasses import dataclass

import dynamiqs as dq

from .simple_system_abc import *
from .param_class import *


@dataclass(kw_only=True)
class ResonatorMode(SimpleSystemABC):
    omega: Param
    omega_rwa: Param = Param(0.0)
    
    
    def Hamiltonian(self):
        H = self.omega * dq.number(self.N)
        H_rwa = self.omega_rwa * dq.number(self.N)
        return H - H_rwa