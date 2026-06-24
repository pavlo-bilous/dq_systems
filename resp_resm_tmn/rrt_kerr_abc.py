from .rrt_abc import *
from ..transmon import *


class RespResmTmnKerrABC(RespResmTmnABC):
    
    def __init__(self,
                 N_res: int,
                 omega_res: float,
                 J: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float,
                 omega_rwa: float
                ):
        
        tmnk = TransmonKerr(
            N=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub,
            omega_rwa=omega_rwa
        )
        super().__init__(N_res, omega_res, omega_rwa, J, tmnk)