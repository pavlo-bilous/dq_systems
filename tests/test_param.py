"""Tests for dq_systems.fundamental.param_class.Param.

Param is the "port" type through which a static-or-time-dependent physical
parameter is injected into a component (see fundamental/param_class.py).
These tests pin down both its numeric duck-typing (static case) and its
special handling of multiplication by a dynamiqs operator (time-dependent
case), including a couple of non-obvious consequences of how it is built
(inheriting numbers.Complex on a frozen dataclass).
"""
import cmath

import dynamiqs as dq
import pytest
from dq_systems.fundamental.param_class import Param, _bad_op_err_msg

# --- static Param: behaves like a plain complex number ---

class TestStaticParam:

    def test_time_dependent_is_false(self):
        assert Param(2.0).time_dependent is False

    def test_call_returns_the_wrapped_value(self):
        assert Param(3 + 4j)(t=123.0) == 3 + 4j

    def test_repr(self):
        assert repr(Param(2.0)) == "Param(2.0)"

    def test_mul_and_rmul_with_plain_number(self):
        assert Param(2.0) * 3 == 6.0
        assert 3 * Param(2.0) == 6.0
        # delegated straight to the wrapped number, not re-wrapped in a Param
        assert type(Param(2.0) * 3) is float

    def test_arithmetic_delegates_to_wrapped_value(self):
        p = Param(3 + 4j)
        assert p + 1 == 4 + 4j
        assert 1 + p == 4 + 4j
        assert -p == -3 - 4j
        assert +p == 3 + 4j
        assert p ** 2 == (3 + 4j) ** 2
        assert p / 2 == (3 + 4j) / 2
        assert 2 / Param(4.0) == 0.5
        assert abs(Param(3 + 4j)) == 5.0
        assert Param(3 + 4j).conjugate() == 3 - 4j
        assert Param(3 + 4j).real == 3.0
        assert Param(3 + 4j).imag == 4.0
        assert complex(Param(3 + 4j)) == 3 + 4j

    def test_two_params_cannot_be_added(self):
        # __add__ delegates to the *wrapped* value's __add__, i.e.
        # (3.0).__add__(Param(1.0)); a plain number does not know how to add
        # a Param, so this correctly fails rather than silently doing
        # something with the outer Param wrapper.
        with pytest.raises(TypeError):
            Param(3.0) + Param(1.0)

    def test_equality_is_dataclass_field_equality_not_delegated(self):
        # Complex.__eq__ is abstract, so in principle it would be filled in
        # by the same delegate-to-`self.p` machinery as the other operators.
        # But `@dataclass(frozen=True)` already generates a concrete __eq__
        # (comparing `(self.p,)` tuples, and only between two Param
        # instances) *before* that machinery runs, so that dataclass-eq
        # wins and is never replaced. The practical consequence: a Param
        # compares equal to another Param wrapping the same value, but NOT
        # to the bare number it wraps.
        assert Param(2.0) == Param(2.0)
        assert Param(2.0) != Param(3.0)
        assert (Param(2.0) == 2.0) is False
        assert (2.0 == Param(2.0)) is False


# --- time-dependent Param: only meaningful op is multiplying a dq operator ---

class TestTimeDependentParam:

    def test_time_dependent_is_true(self):
        assert Param(lambda t: 2.0 * t).time_dependent is True

    def test_call_evaluates_the_callable(self):
        p = Param(lambda t: 2.0 * t + 1.0)
        assert p(0.0) == 1.0
        assert p(2.0) == 5.0

    def test_addition_is_rejected(self):
        p = Param(lambda t: t)
        with pytest.raises(TypeError, match=_bad_op_err_msg.split(":")[0]):
            p + 1

    def test_mul_by_dynamiqs_operator_produces_a_modulated_qarray(self):
        # this is the order used everywhere in the actual dq_systems physics
        # code, e.g. `self.omega * dq.number(self.N)` in ResonatorMode.
        p = Param(lambda t: 2.0 * t)
        op = dq.number(4)
        modulated = p * op

        # at t=0, the modulating function is 0 -> the zero operator
        assert cmath.isclose(modulated(0.0).to_jax()[1, 1].item(), 0.0)
        # at t=3, the modulating function is 6 -> 6 * number(4)
        assert cmath.isclose(modulated(3.0).to_jax()[1, 1].item(), 6.0)
        assert cmath.isclose(modulated(3.0).to_jax()[2, 2].item(), 12.0)

    def test_rmul_does_not_actually_work_with_a_dynamiqs_operand(self):
        # __rmul__ is defined generically as `self * other` (see
        # param_class.py), which is exactly right for `3 * Param(2.0)`
        # (tested under TestStaticParam) since plain numbers don't define
        # __mul__ for a Param and Python falls back to Param.__rmul__.
        # A dynamiqs QArray, however, *does* define __mul__ for arbitrary
        # right-hand operands, and raises NotImplementedError from inside it
        # rather than returning `NotImplemented` -- so Python's reflected-
        # operator fallback to Param.__rmul__ never triggers, and
        # `dynamiqs_operator * time_dependent_param` fails, even though the
        # opposite order (tested above) works fine.
        p = Param(lambda t: 2.0 * t)
        op = dq.number(4)
        with pytest.raises(NotImplementedError):
            op * p
