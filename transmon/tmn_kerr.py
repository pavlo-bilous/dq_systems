import dynamiqs as dq

from .tmn_abc import *
from ..fundamental import *
    

class TransmonKerr(TransmonABC):
    """Transmon in the weakly-anharmonic (Kerr-oscillator) approximation.

    ``H = omega_qub * n + (K / 2) * a^dag a^dag a a``, i.e. a linear
    resonator mode (see :class:`~dq_systems.fundamental.resonator_mode.ResonatorMode`)
    plus a quartic Kerr nonlinearity of strength ``K = -Ec``. This is the
    standard second-order expansion of the full transmon cosine potential,
    valid when ``Ej >> Ec`` (i.e. away from the few-level Cooper-pair-box
    regime).

    Rather than taking ``(Ec, Ej)`` directly like
    :class:`~dq_systems.transmon.tmn_abc.TransmonABC`, this class is
    constructed from the directly-measurable/designed ``(Ec, omega_qub)`` --
    the 0-1 transition frequency -- and derives ``Ej`` internally via
    :meth:`eval_Ej` (the inverse Kerr-approximation relation, see
    :meth:`eval_omega_qub`). ``omega_qub`` may be static or time-dependent
    (e.g. a flux-tuned qubit frequency).
    """

    @staticmethod
    def eval_Ej(Ec, omega_qub):
        """Josephson energy reproducing 0-1 transition frequency ``omega_qub`` at charging energy ``Ec``.

        Inverse of :meth:`eval_omega_qub`, i.e. solves
        ``omega_qub = sqrt(8*Ec*Ej) - Ec`` for ``Ej``.
        """
        return (omega_qub + Ec)**2 / (8 * Ec)


    @staticmethod
    def eval_omega_qub(Ec, Ej):
        """0-1 transition frequency of a transmon with charging/Josephson energies ``(Ec, Ej)``.

        Standard transmon (Kerr-approximation) formula: ``sqrt(8*Ec*Ej) - Ec``.

        KNOWN BUG (not fixed here per instructions): `jnp` is not imported
        into this module's namespace (only `dynamiqs as dq`, and names
        pulled in via `from .tmn_abc import *` / `from ..fundamental import
        *`, neither of which re-exports `jnp`), so calling this raises
        NameError. Not exercised by TransmonKerr.__init__, which only calls
        eval_Ej. See tests/test_transmon.py::TestTransmonKerrEnergyConversion
        for a reproducing (xfail) test. Fix: add `import jax.numpy as jnp`.
        """
        return jnp.sqrt(8 * Ec * Ej) - Ec


    def __init__(self, *,
                 N: int,
                 Ec: float,
                 omega_qub: float | Callable):

        # The linear part of the Hamiltonian is just a resonator at the
        # qubit frequency; reusing ResonatorMode also gives us its Param
        # wrapping of omega_qub for free (static or time-dependent).
        self.linear_part = ResonatorMode(
            N=N,
            omega=omega_qub
        )

        # Convert the user-facing (Ec, omega_qub) parametrization into the
        # (Ec, Ej) one that TransmonABC stores, propagating time-dependence
        # of omega_qub (if any) through to Ej via composition.
        if isinstance(omega_qub, Callable):
            Ej = lambda t: TransmonKerr.eval_Ej(Ec, omega_qub(t))
        else:
            Ej = TransmonKerr.eval_Ej(Ec, omega_qub)

        super().__init__(N, Ec, Ej)


    @property
    def K(self):
        """Kerr (self-)nonlinearity coefficient, ``-Ec``."""
        return -self.Ec


    def Hamiltonian(self):
        H_lin = self.linear_part.Hamiltonian()
        H_nonlin = self.K / 2 * (
            dq.create(self.N) @ dq.create(self.N) @ dq.destroy(self.N) @ dq.destroy(self.N)
        )
        return H_lin + H_nonlin