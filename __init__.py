"""Top-level ``dq_systems`` package.

This is the interfacing layer between upstream physical modeling (e.g. an FEM
simulation reduced to a lumped-element circuit) and the ``dynamiqs`` library,
which is used to simulate the resulting quantum dynamics. Each physical
component (:class:`~dq_systems.fundamental.qsystem_abc.QSystemABC` subclass)
exposes plain constructor arguments as "ports": lumped-element parameters,
either static numbers or time-dependent callables, wrapped in
:class:`~dq_systems.fundamental.param_class.Param`. Every component knows how
to turn its own parameters into a ``dynamiqs`` Hamiltonian via ``.Hamiltonian()``.

Importing this package eagerly pulls in every submodule so all public classes
(``ResonatorMode``, ``TransmonKerr``, ``RespResmTmnKerrDirectDriven``, ...)
are available directly as ``dq_systems.<Name>``.
"""

from .setup import *  # side effect only: configures JAX to x64 precision before anything else runs

from .fundamental import *    # QSystemABC, SimpleSystemABC, CombinedSystem, Param, ResonatorMode
from .transmon import *       # TransmonABC, TransmonKerr
from .resp_resm_tmn import *  # composite resonator(+/-) <-> transmon systems