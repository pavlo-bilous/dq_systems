from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import dynamiqs as dq

from .rrt_kerr_dir import *
from ..fundamental import *


@dataclass
class RespResmTmnKerrDirectDriven(RespResmTmnKerrDirect):
    drive_res: complex | Callable
    drive_tmn: complex | Callable
    omega_drive: float | Callable
    
    
    def __init__(self, *,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 kp_filter: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float | Callable,
                 g: float | Callable,
                 omega_drive: float | Callable,
                 drive_res: complex | Callable,
                 drive_tmn: complex | Callable,
                ):
        self.drive_res = Param(drive_res)
        self.drive_tmn = Param(drive_tmn)
        self.omega_drive = Param(omega_drive)
        
        super().__init__(
            N_res=N_res,
            omega_res=omega_res,
            J=J,
            kp_filter=kp_filter,
            N_tmn=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub,
            g=g,
        )

        
    def V_drive(self):
        N_p, N_m, N_tmn = self.Ns
        
        dop_pm = (
            dq.tensor(dq.create(N_p) + dq.destroy(N_p), dq.eye(N_m), dq.eye(N_tmn)) +
            dq.tensor(dq.eye(N_p), dq.create(N_m) + dq.destroy(N_m), dq.eye(N_tmn))
        )
        Vd_pm = self.drive_res * (dop_pm / (2 * jnp.sqrt(2)))
        
        dop_tmn = dq.tensor(dq.eye(N_p), dq.eye(N_m), dq.create(N_tmn) + dq.destroy(N_tmn))
        Vd_tmn = self.drive_tmn * (dop_tmn / 2)
        
        return Vd_pm + Vd_tmn
    
    
    def H_ref(self):
        num_ops = [dq.number(N) for N in self.Ns]
        num_ops_sum = self.subsys_ops_sum(num_ops)
        return self.omega_drive * num_ops_sum
    
    
    def Hamiltonian(self):
        return super().Hamiltonian() + self.V_drive() - self.H_ref()