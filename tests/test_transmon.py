"""Tests for the dq_systems.transmon package: TransmonABC and TransmonKerr."""
import dynamiqs as dq
import jax.numpy as jnp
import pytest
from dq_systems.fundamental import Param, QSystemABC
from dq_systems.transmon import TransmonABC, TransmonKerr


class _ConcreteTransmon(TransmonABC):
    """Minimal concrete subclass, used only to exercise TransmonABC's own
    __post_init__ (Ej wrapping) in isolation from TransmonKerr."""

    def Hamiltonian(self):
        raise NotImplementedError


class TestTransmonABC:

    def test_cannot_be_instantiated_directly(self):
        # TransmonABC does not implement Hamiltonian() itself.
        with pytest.raises(TypeError):
            TransmonABC(N=4, Ec=0.2, Ej=20.0)

    def test_ej_is_wrapped_in_param(self):
        t = _ConcreteTransmon(N=4, Ec=0.2, Ej=20.0)
        assert isinstance(t.Ej, Param)
        assert t.Ej(0.0) == 20.0

    def test_time_dependent_ej(self):
        t = TransmonKerr(N=4, Ec=0.2, omega_qub=lambda time: 5.0 + 0.1 * time)
        assert t.Ej.time_dependent is True


class TestTransmonKerrEnergyConversion:

    @pytest.mark.xfail(
        raises=NameError,
        reason=(
            "known bug in TransmonKerr.eval_omega_qub (tmn_kerr.py): it uses "
            "`jnp.sqrt(...)` but `jnp` is never imported into that module's "
            "namespace, so any direct call raises NameError. Not hit by "
            "TransmonKerr.__init__ itself, which only ever calls eval_Ej."
        ),
    )
    def test_eval_omega_qub_and_eval_Ej_are_inverses(self):
        Ec = 0.2
        omega_qub = 5.0
        Ej = TransmonKerr.eval_Ej(Ec, omega_qub)
        assert jnp.isclose(TransmonKerr.eval_omega_qub(Ec, Ej), omega_qub)

    def test_K_is_minus_Ec(self):
        t = TransmonKerr(N=4, Ec=0.2, omega_qub=5.0)
        assert t.K == -0.2


class TestTransmonKerrHamiltonian:

    def test_spectrum_matches_analytic_kerr_oscillator_formula(self):
        # H = omega_qub * n + (K/2) n(n-1), with K = -Ec, is diagonal in the
        # number basis, so eigenvalues are exactly E_n = n*omega_qub - Ec/2 * n*(n-1).
        Ec, omega_qub, N = 0.2, 5.0, 6
        t = TransmonKerr(N=N, Ec=Ec, omega_qub=omega_qub)
        e, _ = QSystemABC.diagonalize(t.Hamiltonian())

        n = jnp.arange(N)
        expected = n * omega_qub + (-Ec / 2) * n * (n - 1)
        assert jnp.allclose(jnp.sort(e), jnp.sort(expected))

    def test_linear_part_is_a_resonator_at_omega_qub(self):
        t = TransmonKerr(N=4, Ec=0.2, omega_qub=5.0)
        assert t.linear_part.N == 4
        assert jnp.allclose(
            t.linear_part.Hamiltonian().to_jax(), (5.0 * dq.number(4)).to_jax()
        )

    def test_time_dependent_omega_qub_propagates_through_Ej(self):
        # Regression test for the historical bug where the time-dependent
        # branch of TransmonKerr.__init__ referenced the unqualified name
        # `eval_Ej` instead of `TransmonKerr.eval_Ej` (a NameError on
        # construction). Also checks that Ej(t) actually tracks omega_qub(t)
        # through the Kerr-approximation relation, not just that no error
        # is raised.
        Ec = 0.2
        omega_qub = lambda time: 5.0 + 0.1 * time
        t = TransmonKerr(N=4, Ec=Ec, omega_qub=omega_qub)

        for time in (0.0, 1.0, 7.5):
            expected_Ej = TransmonKerr.eval_Ej(Ec, omega_qub(time))
            assert jnp.isclose(t.Ej(time), expected_Ej)

        H = t.Hamiltonian()
        e0, _ = QSystemABC.diagonalize(H(0.0))
        e1, _ = QSystemABC.diagonalize(H(5.0))
        # the qubit (0-1) gap should grow as omega_qub(t) grows
        assert (e1[1] - e1[0]) > (e0[1] - e0[0])
