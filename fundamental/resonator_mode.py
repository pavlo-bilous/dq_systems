from dataclasses import dataclass

import dynamiqs as dq

from .simple_system_abc import *
from .param_class import *


@dataclass(kw_only=True)
class ResonatorMode(SimpleSystemABC):
    """A single harmonic oscillator mode, i.e. ``H = omega * n``.

    The simplest possible :class:`~dq_systems.fundamental.simple_system_abc.SimpleSystemABC`:
    a linear resonator/cavity mode of angular frequency ``omega``, truncated
    to ``N`` Fock states. Also reused as the linear part of other systems,
    e.g. :class:`~dq_systems.transmon.tmn_kerr.TransmonKerr`.
    """

    omega: float | Callable  #: mode frequency; static value or a callable omega(t) for a time-dependent frequency


    def __post_init__(self):
        # Wrap the raw frequency into a Param "port" (see param_class.py) so
        # Hamiltonian() below works uniformly whether omega is a number or a
        # time-dependent callable.
        self.omega = Param(self.omega)


    def Hamiltonian(self):
        return self.omega * dq.number(self.N)