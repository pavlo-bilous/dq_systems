from collections import namedtuple
from abc import ABC, abstractmethod

from ..fundamental import *
from ..transmon import *


class RespResmTmnABC(CombinedSystem):
    
    def __init__(self,
                 N_res: int,
                 omega_res: float,
                 omega_res_rwa: float,
                 J: float,
                 tmn: TransmonABC
                ):
        
        Subsystems = namedtuple('Subsystems', ['res_p', 'res_m', 'tmn'])
        
        subsystems = Subsystems(
            res_p=ResonatorMode(N=N_res, omega=omega_res+J, omega_rwa=omega_res_rwa),
            res_m=ResonatorMode(N=N_res, omega=omega_res-J, omega_rwa=omega_res_rwa),
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