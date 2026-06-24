from collections.abc import Callable

from .rrt_abc import *
from ..transmon import *


class RespResmTmnKerrABC(RespResmTmnABC):
    
    def __init__(self,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float | Callable
                ):
        
        tmnk = TransmonKerr(
            N=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub
        )
        super().__init__(N_res, omega_res, J, tmnk)