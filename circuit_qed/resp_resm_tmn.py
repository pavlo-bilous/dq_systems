from dataclasses import dataclass
from collections.abc import Callable
from collections import namedtuple

import jax.numpy as jnp
import dynamiqs as dq

from ..general import *
from .components import *
    

@dataclass
class RespResmTmn(CombinedSystem):
    g: float
    
    def __init__(self,
                 *,
                 N_res: int,
                 omega_res: float,
                 J: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float,
                 g: float
                ):
        self.g = g
        
        Subsystems = namedtuple('Subsystems', ['res_p', 'res_m', 'tmn'])
        
        subsystems = Subsystems(
            res_p=Resonator(N=N_res, omega=omega_res+J),
            res_m=Resonator(N=N_res, omega=omega_res-J),
            tmn=TransmonKerr(N=N_tmn, Ec=Ec, omega=omega_qub)
        )
        
        super().__init__(subsystems)
    
    
    def V_interact(self):
        N_p, N_m, N_tmn = self.Ns
        vr = dq.tensor(dq.create(N_p), dq.eye(N_m))
        vr += dq.tensor(dq.eye(N_p), dq.create(N_m))
        V = self.g / jnp.sqrt(2) * dq.tensor(vr, dq.destroy(N_tmn))
        return V + V.dag()
    
    
    def H_nointeract(self):
        return super().Hamiltonian()
    
    
    def Hamiltonian(self):
        return self.H_nointeract() + self.V_interact()
        
    
    
@dataclass
class RespResmTmnDrivenRF(RespResmTmn):
    omega_drive: float
    fdrive_tmn: Callable
    fdrive_res: Callable
    
    def __init__(self,
                 *,
                 N_res: int,
                 omega_res: float,
                 J: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float,
                 g: float,
                 omega_drive: float,
                 fdrive_tmn: Callable,
                 fdrive_res: Callable
                ):
        
        self.omega_drive = omega_drive
        self.fdrive_tmn = fdrive_tmn
        self.fdrive_res = fdrive_res
        
        super().__init__(
            N_res=N_res,
            omega_res=omega_res,
            J=J,
            N_tmn=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub,
            g=g
        )    
        
        
    def H_ref(self):
        N_ops = [dq.number(N) for N in self.Ns]
        N_op_sum = self.subsys_ops_sum(N_ops)
        return self.omega_drive * N_op_sum
    
    
    def H_nointeract(self):
        return super().H_nointeract() - self.H_ref()
    
    
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