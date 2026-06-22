from dataclasses import dataclass

import dynamiqs as dq

from ...general import *


@dataclass(kw_only=True)
class Resonator(SimpleSystem):
    omega: float
    
    def Hamiltonian(self):
        return self.omega * dq.number(self.N)