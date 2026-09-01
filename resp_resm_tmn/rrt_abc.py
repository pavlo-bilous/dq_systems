from collections import namedtuple
from collections.abc import Callable
from abc import ABC, abstractmethod

from ..fundamental import *
from ..transmon import *


class RespResmTmnABC(CombinedSystem):
    """Abstract base for a resonator (split into +/- normal modes) coupled to a transmon.

    Models a readout resonator that hybridizes into two normal modes
    ``res_p``/``res_m`` (e.g. from being coupled to a second resonator or a
    Purcell filter), symmetrically split about ``omega_res`` by coupling
    strength ``J`` (``omega_p = omega_res + J``, ``omega_m = omega_res - J``),
    together with a transmon subsystem ``tmn``. Subsystems are exposed both
    positionally (via :attr:`CombinedSystem.subsystems`, inherited) and by
    name, since ``subsystems`` is a namedtuple with fields ``res_p``,
    ``res_m``, ``tmn``.

    Concrete subclasses must supply the interaction Hamiltonian
    (:meth:`V_interact`, added on top of the bare
    :meth:`~dq_systems.fundamental.combined_system.CombinedSystem.Hamiltonian`
    in :meth:`Hamiltonian`) and the collapse/relaxation operators
    (:meth:`relax_ops`) for open-system (Lindblad) simulation.
    """

    def __init__(self,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 tmn: TransmonABC
                ):

        Subsystems = namedtuple('Subsystems', ['res_p', 'res_m', 'tmn'])

        # omega_res is the (uncoupled) resonator frequency; the two
        # hybridized normal modes sit symmetrically at omega_res +/- J.
        # Time-dependence of omega_res (if any) is propagated to both modes.
        if isinstance(omega_res, Callable):
            omega_p = lambda t: omega_res(t) + J
            omega_m = lambda t: omega_res(t) - J
        else:
            omega_p = omega_res + J
            omega_m = omega_res - J

        subsystems = Subsystems(
            res_p=ResonatorMode(N=N_res, omega=omega_p),
            res_m=ResonatorMode(N=N_res, omega=omega_m),
            tmn=tmn
        )
        super().__init__(subsystems)


    @abstractmethod
    def V_interact(self):
        """Interaction Hamiltonian coupling the resonator modes and the transmon."""
        pass


    @abstractmethod
    def relax_ops(self):
        """List of collapse ("jump") operators describing dissipation, for Lindblad-form simulation."""
        pass


    def Hamiltonian(self):
        return super().Hamiltonian() + self.V_interact()