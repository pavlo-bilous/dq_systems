from dataclasses import dataclass

import jax.numpy as jnp
import dynamiqs as dq

from .rrt_kerr_abc import *
from ..fundamental import *


@dataclass
class RespResmTmnKerrDirect(RespResmTmnKerrABC):
    kp_filter: float
    g: float | Callable
    
    def __init__(self, *,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 kp_filter: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float | Callable,
                 g: float | Callable
                ):
        self.kp_filter = kp_filter
        self.g = Param(g)
        
        super().__init__(N_res, omega_res, J,
                 N_tmn, Ec, omega_qub
        )
        
        
    def V_interact(self):
        N_p, N_m, N_tmn = self.Ns
        vr = dq.tensor(dq.create(N_p), dq.eye(N_m))
        vr += dq.tensor(dq.eye(N_p), dq.create(N_m))
        V = self.g * (dq.tensor(vr, dq.destroy(N_tmn)) / jnp.sqrt(2))
        return V + V.dag()
    
    
    def relax_ops(self):
        N_p, N_m, N_tmn = self.Ns
        a_p = dq.tensor(dq.destroy(N_p), dq.eye(N_m))
        a_m = dq.tensor(dq.eye(N_p), dq.destroy(N_m))
        rop = dq.tensor(a_p - a_m, dq.eye(N_tmn))
        c_relax = jnp.sqrt(self.kp_filter / 2) * rop
        return [c_relax]