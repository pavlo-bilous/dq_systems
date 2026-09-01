from dataclasses import dataclass
from collections.abc import Callable

from ..fundamental import *


@dataclass
class TransmonABC(SimpleSystemABC):
    """Base class for a transmon qubit, parametrized by its two circuit energies.

    ``Ec`` is the charging energy and ``Ej`` the Josephson energy of the
    transmon, in the same angular-frequency units as the rest of the model
    (:math:`\\hbar = 1`). Concrete subclasses (e.g.
    :class:`~dq_systems.transmon.tmn_kerr.TransmonKerr`) decide how ``Ec``/
    ``Ej`` translate into an actual :meth:`~dq_systems.fundamental.qsystem_abc.QSystemABC.Hamiltonian`.
    """

    Ec: float  #: charging energy (static; the Kerr/anharmonicity scale for TransmonKerr)
    Ej: float | Callable  #: Josephson energy; static value or a callable Ej(t) for flux-tunable transmons


    def __post_init__(self):
        # Wrap Ej into a Param "port" so it can transparently be static or
        # time-dependent (flux-tunable transmon), same pattern as ResonatorMode.omega.
        self.Ej = Param(self.Ej)