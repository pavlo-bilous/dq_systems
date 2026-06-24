from collections import namedtuple
from collections.abc import Callable
from abc import ABC, abstractmethod

from ..fundamental import *
from ..transmon import *


class RespResmTmnABC(CombinedSystem):
    
    def __init__(self,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 tmn: TransmonABC
                ):
        
        Subsystems = namedtuple('Subsystems', ['res_p', 'res_m', 'tmn'])

        if isinstance(omega_res, Callable):
            omega_p = lambda t: omega_res(t) + J
            omega_m = lambda t: omega_res(t) - J
        else:
            omega_p = omega_res + J
            omega_m = omega_res - J    
        
        subsystems = Subsystems(
            res_p=ResonatorMode(N=N_res, omega=omega_p),
            res_m=ResonatorMode(N=N_res, omega=omega_m),
            tmn=tmn
        )
        super().__init__(subsystems)
    

    @abstractmethod
    def V_interact(self):
        pass
    
    
    @abstractmethod
    def relax_ops(self):
        pass
    

    def Hamiltonian(self):
        return super().Hamiltonian() + self.V_interact()