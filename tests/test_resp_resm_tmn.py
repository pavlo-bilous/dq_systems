"""Tests for the dq_systems.resp_resm_tmn package: the resonator(+/-)-transmon
composite systems, in their increasingly concrete layers.
"""
import dynamiqs as dq
import jax.numpy as jnp
import pytest
from dq_systems.resp_resm_tmn import (
    RespResmTmnABC,
    RespResmTmnKerrABC,
    RespResmTmnKerrDirect,
    RespResmTmnKerrDirectDriven,
)
from dq_systems.transmon import TransmonKerr

N_RES, OMEGA_RES, J = 3, 6.0, 0.05
N_TMN, EC, OMEGA_QUB = 4, 0.2, 5.0
KP_FILTER, G = 0.01, 0.1


def make_direct(**overrides):
    kwargs = {
        "N_res": N_RES, "omega_res": OMEGA_RES, "J": J, "kp_filter": KP_FILTER,
        "N_tmn": N_TMN, "Ec": EC, "omega_qub": OMEGA_QUB, "g": G,
    }
    kwargs.update(overrides)
    return RespResmTmnKerrDirect(**kwargs)


def make_driven(**overrides):
    kwargs = {
        "N_res": N_RES, "omega_res": OMEGA_RES, "J": J, "kp_filter": KP_FILTER,
        "N_tmn": N_TMN, "Ec": EC, "omega_qub": OMEGA_QUB, "g": G,
        "omega_drive": OMEGA_RES, "drive_res": 0.0, "drive_tmn": 0.0,
    }
    kwargs.update(overrides)
    return RespResmTmnKerrDirectDriven(**kwargs)


def is_hermitian(H):
    m = H.to_jax()
    return jnp.allclose(m, m.conj().T)


class TestAbstractLayers:

    def test_rrt_abc_cannot_be_instantiated(self):
        tmn = TransmonKerr(N=N_TMN, Ec=EC, omega_qub=OMEGA_QUB)
        with pytest.raises(TypeError):
            RespResmTmnABC(N_res=N_RES, omega_res=OMEGA_RES, J=J, tmn=tmn)

    def test_rrt_kerr_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            RespResmTmnKerrABC(
                N_res=N_RES, omega_res=OMEGA_RES, J=J,
                N_tmn=N_TMN, Ec=EC, omega_qub=OMEGA_QUB,
            )


class TestRespResmTmnKerrDirect:

    def test_normal_modes_split_symmetrically_around_omega_res(self):
        sys_ = make_direct()
        assert sys_.subsystems.res_p.omega(0.0) == OMEGA_RES + J
        assert sys_.subsystems.res_m.omega(0.0) == OMEGA_RES - J

    def test_Ns_matches_res_p_res_m_tmn_order(self):
        sys_ = make_direct()
        assert sys_.Ns == [N_RES, N_RES, N_TMN]

    def test_hamiltonian_shape_and_hermiticity(self):
        sys_ = make_direct()
        H = sys_.Hamiltonian()
        assert H.dims == (N_RES, N_RES, N_TMN)
        assert is_hermitian(H)

    def test_hamiltonian_is_bare_plus_interaction(self):
        sys_ = make_direct()
        # CombinedSystem.Hamiltonian (bare, uncoupled) + V_interact, per
        # RespResmTmnABC.Hamiltonian.
        from dq_systems.fundamental import CombinedSystem
        bare = CombinedSystem.Hamiltonian(sys_)
        expected = bare + sys_.V_interact()
        assert jnp.allclose(sys_.Hamiltonian().to_jax(), expected.to_jax())

    def test_relax_ops_rate_scaling(self):
        sys_ = make_direct(kp_filter=0.04)
        [c] = sys_.relax_ops()
        a_p = dq.tensor(dq.destroy(N_RES), dq.eye(N_RES), dq.eye(N_TMN))
        a_m = dq.tensor(dq.eye(N_RES), dq.destroy(N_RES), dq.eye(N_TMN))
        expected = jnp.sqrt(0.04 / 2) * (a_p - a_m).to_jax()
        assert jnp.allclose(c.to_jax(), expected)

    def test_time_dependent_coupling_g(self):
        sys_ = make_direct(g=lambda t: 0.1 * jnp.cos(t))
        H = sys_.Hamiltonian()
        # at t=0, g(0)=0.1 matches the static-g Hamiltonian exactly
        static = make_direct(g=0.1).Hamiltonian()
        assert jnp.allclose(H(0.0).to_jax(), static.to_jax())


class TestRespResmTmnKerrDirectDriven:

    def test_hamiltonian_equals_direct_plus_drive_minus_reference(self):
        sys_ = make_driven(drive_res=0.02, drive_tmn=0.01)
        # RespResmTmnKerrDirect.Hamiltonian via super(), independent of any
        # drive attributes (structural/composition check).
        base = RespResmTmnKerrDirect.Hamiltonian(sys_)
        expected = base + sys_.V_drive() - sys_.H_ref()
        assert jnp.allclose(sys_.Hamiltonian().to_jax(), expected.to_jax())

    def test_static_drive_hamiltonian_is_hermitian(self):
        sys_ = make_driven(drive_res=0.02, drive_tmn=0.01)
        assert is_hermitian(sys_.Hamiltonian())

    def test_time_dependent_drive_matches_frozen_time_reconstruction(self):
        # A time-dependent drive_res, evaluated at a given t, should agree
        # with a wholly-static system built with drive_res fixed to that
        # same value at that t -- i.e. the Param "port" mechanism correctly
        # threads time-dependence through the whole Hamiltonian assembly.
        drive_fn = lambda t: 0.02 * jnp.cos(t)
        dynamic = make_driven(drive_res=drive_fn)
        for t in (0.0, 1.3, 4.0):
            static = make_driven(drive_res=float(drive_fn(t)))
            assert jnp.allclose(
                dynamic.Hamiltonian()(t).to_jax(),
                static.Hamiltonian().to_jax(),
            )

    def test_H_ref_is_omega_drive_times_total_number_operator(self):
        sys_ = make_driven(omega_drive=6.0)
        num_p = dq.tensor(dq.number(N_RES), dq.eye(N_RES), dq.eye(N_TMN))
        num_m = dq.tensor(dq.eye(N_RES), dq.number(N_RES), dq.eye(N_TMN))
        num_t = dq.tensor(dq.eye(N_RES), dq.eye(N_RES), dq.number(N_TMN))
        expected = 6.0 * (num_p + num_m + num_t)
        assert jnp.allclose(sys_.H_ref().to_jax(), expected.to_jax())
