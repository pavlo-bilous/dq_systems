from abc import ABC, abstractmethod
import inspect

import jax.numpy as jnp
import dynamiqs as dq


class QSystemABC(ABC):
    """Root abstraction for every modeled quantum system in dq_systems.

    A ``QSystemABC`` is anything that can produce a ``dynamiqs`` Hamiltonian
    for itself via :meth:`Hamiltonian`. Concrete subclasses range from a
    single physical mode (see :class:`~dq_systems.fundamental.simple_system_abc.SimpleSystemABC`)
    up to full composite circuits (see :class:`~dq_systems.fundamental.combined_system.CombinedSystem`
    and the ``resp_resm_tmn`` package).
    """

    @abstractmethod
    def Hamiltonian(self):
        """Return this system's Hamiltonian as a ``dynamiqs`` (qarray-like) object.

        The result may be time-independent or, if any of the system's
        :class:`~dq_systems.fundamental.param_class.Param` are time-dependent,
        a ``dynamiqs`` time-dependent array (e.g. ``ModulatedTimeQArray`` /
        ``SummedTimeQArray``) directly consumable by dynamiqs solvers.
        """
        pass


    @staticmethod
    def diagonalize(H):
        """Exactly diagonalize a (time-independent) Hermitian Hamiltonian ``H``.

        Returns ``(e, v)`` where ``e`` are the eigenenergies (ascending) and
        ``v`` is a ``dynamiqs`` qarray stacking the corresponding eigenstates
        (as kets) along its leading axis, each carrying ``H``'s Hilbert-space
        ``dims`` -- e.g. useful for reading off dressed-state energies of a
        subsystem or a full composite system.
        """
        e, v = jnp.linalg.eigh(H.to_jax())
        v = v.T[:, :, jnp.newaxis]
        return e, dq.asqarray(v, dims=H.dims)


    @classmethod
    def init_from_dict(cls, d: dict):
        """Construct ``cls`` from a dict of named parameters, ignoring extra keys.

        Convenience for wiring up a system straight from an upstream
        lumped-element-extraction result (e.g. a dict of many circuit
        parameters), without having to manually pick out the subset that a
        given system's constructor actually accepts.
        """
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters) - {"self"}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)