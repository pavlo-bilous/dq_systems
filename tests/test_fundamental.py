"""Tests for the dq_systems.fundamental package: QSystemABC, SimpleSystemABC,
ResonatorMode and CombinedSystem.
"""
import dynamiqs as dq
import jax.numpy as jnp
import pytest
from dq_systems.fundamental import (
    CombinedSystem,
    Param,
    QSystemABC,
    ResonatorMode,
    SimpleSystemABC,
)


class TestSimpleSystemABC:

    def test_cannot_be_instantiated_directly(self):
        # Hamiltonian() is abstract (inherited from QSystemABC) and
        # SimpleSystemABC does not implement it.
        with pytest.raises(TypeError):
            SimpleSystemABC(N=3)


class TestQSystemABCHelpers:

    def test_diagonalize_returns_ascending_eigenvalues_and_matching_dims(self):
        r = ResonatorMode(N=4, omega=5.0)
        H = r.Hamiltonian()
        e, v = QSystemABC.diagonalize(H)

        assert jnp.allclose(e, jnp.array([0.0, 5.0, 10.0, 15.0]))
        assert v.dims == H.dims
        # one eigenvector (as a ket) per eigenvalue
        assert v.shape[0] == 4

    def test_init_from_dict_filters_unknown_keys(self):
        r = ResonatorMode.init_from_dict({"N": 4, "omega": 5.0, "extra_from_fem": 123})
        assert r.N == 4
        assert r.omega(0.0) == 5.0

    def test_init_from_dict_still_requires_mandatory_keys(self):
        with pytest.raises(TypeError):
            ResonatorMode.init_from_dict({"N": 4})


class TestResonatorMode:

    def test_hamiltonian_static_omega(self):
        r = ResonatorMode(N=4, omega=5.0)
        H = r.Hamiltonian()
        assert jnp.allclose(H.to_jax(), (5.0 * dq.number(4)).to_jax())

    def test_omega_is_wrapped_in_param(self):
        r = ResonatorMode(N=4, omega=5.0)
        assert isinstance(r.omega, Param)
        assert r.omega.time_dependent is False

    def test_hamiltonian_time_dependent_omega(self):
        r = ResonatorMode(N=3, omega=lambda t: 5.0 + 0.1 * t)
        H = r.Hamiltonian()
        assert jnp.allclose(H(0.0).to_jax(), (5.0 * dq.number(3)).to_jax())
        assert jnp.allclose(H(2.0).to_jax(), (5.2 * dq.number(3)).to_jax())


class TestCombinedSystem:

    def _two_resonators(self):
        return CombinedSystem(subsystems=[
            ResonatorMode(N=3, omega=5.0),
            ResonatorMode(N=2, omega=7.0),
        ])

    def test_Ns(self):
        cs = self._two_resonators()
        assert cs.Ns == [3, 2]

    def test_hamiltonian_is_tensor_sum_of_embedded_subsystem_hamiltonians(self):
        cs = self._two_resonators()
        H = cs.Hamiltonian()

        expected = dq.tensor(5.0 * dq.number(3), dq.eye(2)) + dq.tensor(dq.eye(3), 7.0 * dq.number(2))
        assert H.dims == (3, 2)
        assert jnp.allclose(H.to_jax(), expected.to_jax())

    def test_subsys_ops_sum_with_three_subsystems(self):
        cs = CombinedSystem(subsystems=[
            ResonatorMode(N=2, omega=1.0),
            ResonatorMode(N=2, omega=1.0),
            ResonatorMode(N=2, omega=1.0),
        ])
        ops = [dq.number(2), dq.number(2), dq.number(2)]
        total = cs.subsys_ops_sum(ops)

        expected = (
            dq.tensor(dq.number(2), dq.eye(2), dq.eye(2))
            + dq.tensor(dq.eye(2), dq.number(2), dq.eye(2))
            + dq.tensor(dq.eye(2), dq.eye(2), dq.number(2))
        )
        assert jnp.allclose(total.to_jax(), expected.to_jax())
