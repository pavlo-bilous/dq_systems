from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import dynamiqs as dq

from .rrt_kerr_dir import *


@dataclass
class RespResmTmnKerrDirectDriven(RespResmTmnKerrDirect):
    fdrive_res: Callable
    fdrive_tmn: Callable
    
    
    def __init__(self, *,
                 N_res: int,
                 omega_res: float,
                 J: float,
                 kp_filter: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float,
                 g: float,
                 omega_drive: float,
                 fdrive_res: Callable,
                 fdrive_tmn: Callable,
                ):
        self.fdrive_res = fdrive_res
        self.fdrive_tmn = fdrive_tmn
        
        super().__init__(
            N_res=N_res,
            omega_res=omega_res,
            J=J,
            kp_filter=kp_filter,
            N_tmn=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub,
            g=g,
            omega_rwa=omega_drive
        )
        
        
    def V_drive(self):
        N_p, N_m, N_tmn = self.Ns
        
        dop_pm = (
            dq.tensor(dq.create(N_p) + dq.destroy(N_p), dq.eye(N_m), dq.eye(N_tmn)) +
            dq.tensor(dq.eye(N_p), dq.create(N_m) + dq.destroy(N_m), dq.eye(N_tmn))
        )
        Vd_pm = dq.modulated(
            self.fdrive_res,
            1 / (2 * jnp.sqrt(2)) * dop_pm
        )
        
        dop_tmn = dq.tensor(dq.eye(N_p), dq.eye(N_m), dq.create(N_tmn) + dq.destroy(N_tmn))
        Vd_tmn = dq.modulated(
            self.fdrive_tmn,
            1 / 2 * dop_tmn
        )
        
        return Vd_pm + Vd_tmn
    
    
    def Hamiltonian(self):
        return super().Hamiltonian() + self.V_drive()