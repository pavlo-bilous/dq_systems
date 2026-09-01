from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import dynamiqs as dq

from .rrt_kerr_dir import *
from ..fundamental import *


@dataclass
class RespResmTmnKerrDirectDriven(RespResmTmnKerrDirect):
    """:class:`~dq_systems.resp_resm_tmn.rrt_kerr_dir.RespResmTmnKerrDirect` with external drives, in the drive's rotating frame.

    Adds coherent drive terms on the resonator (``drive_res``) and on the
    transmon (``drive_tmn``), and moves the whole Hamiltonian into the frame
    rotating at ``omega_drive`` by subtracting the reference Hamiltonian
    :meth:`H_ref` (``omega_drive`` times the total excitation-number
    operator) -- the standard "drive frame" used so that, on resonance, drive
    terms become quasi-static rather than oscillating at optical frequencies.
    """

    drive_res: complex | Callable  #: resonator drive amplitude; static or time-dependent (e.g. a pulse envelope)
    drive_tmn: complex | Callable  #: transmon drive amplitude; static or time-dependent
    omega_drive: float | Callable  #: rotating-frame (drive) frequency; static or time-dependent


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
        """Coherent drive Hamiltonian on the resonator (symmetric p/m combination) and the transmon."""
        N_p, N_m, N_tmn = self.Ns

        # Drive couples to the symmetric (original, un-hybridized) resonator
        # mode combination, same (a_p + a_m)/sqrt(2) recombination as
        # V_interact in RespResmTmnKerrDirect; here it's (x + x^dag)-like
        # (create + destroy) since a drive term is real/Hermitian by itself,
        # with the complex drive amplitude carried by the Param.
        dop_pm = (
            dq.tensor(dq.create(N_p) + dq.destroy(N_p), dq.eye(N_m), dq.eye(N_tmn)) +
            dq.tensor(dq.eye(N_p), dq.create(N_m) + dq.destroy(N_m), dq.eye(N_tmn))
        )
        Vd_pm = self.drive_res * (dop_pm / (2 * jnp.sqrt(2)))

        dop_tmn = dq.tensor(dq.eye(N_p), dq.eye(N_m), dq.create(N_tmn) + dq.destroy(N_tmn))
        Vd_tmn = self.drive_tmn * (dop_tmn / 2)

        return Vd_pm + Vd_tmn


    def H_ref(self):
        """Reference Hamiltonian defining the rotating frame: ``omega_drive`` times the total number operator."""
        num_ops = [dq.number(N) for N in self.Ns]
        num_ops_sum = self.subsys_ops_sum(num_ops)
        return self.omega_drive * num_ops_sum


    def Hamiltonian(self):
        # Lab-frame bare + interaction Hamiltonian (from RespResmTmnKerrDirect),
        # plus the drive, minus the rotating-frame reference.
        return super().Hamiltonian() + self.V_drive() - self.H_ref()