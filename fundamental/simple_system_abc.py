from dataclasses import dataclass

from .qsystem_abc import *


@dataclass
class SimpleSystemABC(QSystemABC):
    """Base class for a single "atomic" quantum system (one Hilbert-space factor).

    Subclasses (e.g. :class:`~dq_systems.fundamental.resonator_mode.ResonatorMode`,
    :class:`~dq_systems.transmon.tmn_abc.TransmonABC`) represent one physical
    mode/degree of freedom, truncated to a Fock-space dimension ``N``, and are
    the leaves that :class:`~dq_systems.fundamental.combined_system.CombinedSystem`
    tensors together into a composite system.
    """

    N: int  #: Hilbert-space (Fock) truncation dimension for this mode.