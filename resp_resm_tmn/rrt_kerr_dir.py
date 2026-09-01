from dataclasses import dataclass

import jax.numpy as jnp
import dynamiqs as dq

from .rrt_kerr_abc import *
from ..fundamental import *


@dataclass
class RespResmTmnKerrDirect(RespResmTmnKerrABC):
    """Resonator(+/-)-transmon system with direct (beam-splitter) coupling and filter relaxation.

    Adds two things on top of :class:`~dq_systems.resp_resm_tmn.rrt_kerr_abc.RespResmTmnKerrABC`:

    - :meth:`V_interact`: a Jaynes-Cummings-like (rotating-wave, beam-splitter)
      coupling of strength ``g`` between the transmon and the *symmetric*
      combination of the two resonator normal modes, ``(a_p + a_m)/sqrt(2)``
      -- i.e. direct coupling to the original (un-hybridized) resonator mode.
    - :meth:`relax_ops`: a single collapse operator on the *antisymmetric*
      combination ``(a_p - a_m)``, at rate ``kp_filter``, modeling loss
      through a Purcell filter (whichever normal-mode combination the filter
      couples to).
    """

    kp_filter: float  #: relaxation rate (through the filter channel) of the antisymmetric resonator combination
    g: float | Callable  #: resonator-transmon coupling strength; static or time-dependent

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
        # vr = a_p^dag + a_m^dag acting on the two resonator-mode factors
        # (identity on the transmon factor, added in via dq.tensor below);
        # dividing by sqrt(2) recombines the p/m normal modes back into the
        # original resonator mode's creation operator.
        N_p, N_m, N_tmn = self.Ns
        vr = dq.tensor(dq.create(N_p), dq.eye(N_m))
        vr += dq.tensor(dq.eye(N_p), dq.create(N_m))
        V = self.g * (dq.tensor(vr, dq.destroy(N_tmn)) / jnp.sqrt(2))
        return V + V.dag()  # Hermitian: g*(vr^dag_res * a_tmn) + h.c.


    def relax_ops(self):
        # a_p - a_m is the antisymmetric normal-mode combination; this is
        # the channel through which the Purcell filter drains excitations,
        # at rate kp_filter (the sqrt(.../2) normalization matches the
        # Lindblad convention where the dissipator rate is |c|^2).
        N_p, N_m, N_tmn = self.Ns
        a_p = dq.tensor(dq.destroy(N_p), dq.eye(N_m))
        a_m = dq.tensor(dq.eye(N_p), dq.destroy(N_m))
        rop = dq.tensor(a_p - a_m, dq.eye(N_tmn))
        c_relax = jnp.sqrt(self.kp_filter / 2) * rop
        return [c_relax]