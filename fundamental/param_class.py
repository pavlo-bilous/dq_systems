from dataclasses import dataclass
from collections.abc import Callable
from numbers import Complex

import dynamiqs as dq


# Raised whenever a time-dependent Param is used in an arithmetic op other
# than multiplication by a dynamiqs operator -- e.g. adding two time-dependent
# Params, or diagonalizing a time-dependent Hamiltonian, has no well-defined
# meaning here and is deliberately not supported.
_bad_op_err_msg = "Bad operation: Time-dependend parameters can only be multiplied with dynamiqs objects."


@dataclass(frozen=True)
class Param(Complex):
    """The "port" type: a physical parameter that may be static or time-dependent.

    This is the interface through which lumped-element parameters (a
    frequency, a coupling strength, a drive amplitude, ...) coming from a
    lower modeling layer are injected into a component. A ``Param`` wraps
    either a plain complex number, or a ``Callable[[float], complex]`` giving
    the parameter's value at time ``t`` (see :attr:`time_dependent`).

    Subclassing ``numbers.Complex`` lets a *static* ``Param`` be used as a
    drop-in replacement for a plain number anywhere in the surrounding
    physics code (arithmetic, comparisons, ``float()``/``complex()``, ...) --
    all such operations are delegated to the wrapped value ``p`` (see
    ``_delegate_op`` and ``_complete_ops`` below). A *time-dependent* Param
    only supports being multiplied by a dynamiqs operator (``__mul__``/
    ``__rmul__``), producing a ``dynamiqs`` time-modulated array; any other
    operation on a time-dependent Param raises ``TypeError``, since e.g.
    "evaluate this callable-valued number without a time" is undefined.
    """

    p: complex | Callable[[float], complex]

    def __repr__(self):
        return f"Param({self.p})"


    @property
    def time_dependent(self) -> bool:
        """Whether this Param wraps a callable (time-dependent) rather than a plain number."""
        return isinstance(self.p, Callable)


    def __call__(self, t: float):
        """Evaluate the parameter at time ``t`` (a no-op if it is static)."""
        if self.time_dependent:
            return self.p(t)
        else:
            return self.p


    def __mul__(self, other):
        # Static case: behaves like ordinary scalar multiplication (delegated
        # below via _complete_ops for the other arithmetic dunders).
        # Time-dependent case: only multiplication by a dynamiqs operator is
        # meaningful -- it produces a dynamiqs "modulated" time-dependent
        # array, i.e. this is where a time-dependent physical parameter turns
        # into a time-dependent Hamiltonian/operator term for dynamiqs.
        if self.time_dependent:
            if isinstance(other, dq.QArrayLike):
                return dq.modulated(self.p, other)
            else:
                raise TypeError(_bad_op_err_msg)
        else:
            return self.p * other


    def __rmul__(self, other):
        return self * other
    
    
    
# --- Metaprogramming: fill in the rest of the numbers.Complex interface ---
#
# Complex requires quite a few abstract dunders (__add__, __radd__, __neg__,
# __pos__, __truediv__, __rtruediv__, __pow__, __rpow__, __abs__, conjugate,
# __eq__, real, imag, ...). __mul__/__rmul__ are handled explicitly above
# (they need the time-dependent/dynamiqs special-casing); every other
# required method is generated here and simply forwards to the same-named
# operation on the wrapped value `self.p`, so a *static* Param behaves
# exactly like the plain number it wraps. For a *time-dependent* Param
# (self.p is a callable), `_delegate_op` refuses instead, since e.g. "add
# two functions of time as if they were numbers" isn't a defined operation
# for this class -- only the explicit __mul__/__rmul__ above know how to
# handle the time-dependent case.

def _delegate_op(self, op_name, *args, **kwargs):
    if isinstance(self.p, Callable):
        raise TypeError(_bad_op_err_msg)
    op = getattr(self.p, op_name)
    if op_name in "imag real".split():
        res = op
    else:
        res = op(*args, **kwargs)
    return res


def _write_op_str(op_name):
    # Builds the source of a small wrapper method `def <op_name>(self, ...)`
    # that calls _delegate_op; `real`/`imag` are properties on plain numbers
    # rather than methods, so they get a @property wrapper instead.
    op = (
f"""def {op_name}(self, *args, **kwargs):
    return _delegate_op(self, '{op_name}', *args, **kwargs)"""
    )
    if op_name in "imag real".split():
        op = "@property\n" + op
    return op


def _complete_ops():
    # Iterates exactly the abstract methods that Complex still demands (i.e.
    # everything not already given a concrete definition above, such as
    # __mul__/__rmul__), generates+execs a delegating implementation for
    # each, attaches them to the class, and finally clears
    # __abstractmethods__ so Param becomes instantiable.
    #
    # NOTE on __eq__: `@dataclass(frozen=True)` (see the class definition
    # above) already generates a concrete __eq__ that compares `(self.p,)`
    # tuples, and only between two Param instances (NotImplemented
    # otherwise) -- so by the time this function runs, '__eq__' is no
    # longer in Param.__abstractmethods__ and never gets the delegating
    # implementation below. Practical effect: `Param(2.0) == Param(2.0)` is
    # True, but `Param(2.0) == 2.0` is False (delegation would have made it
    # True, matching plain-number semantics -- this is a case where the
    # dataclass machinery silently wins over the Complex-delegation intent).
    ops = {}

    for op_name in Param.__abstractmethods__:
        op_str = _write_op_str(op_name)
        exec(op_str, {'_delegate_op': _delegate_op}, ops)

    for op_name, op in ops.items():
        setattr(Param, op_name, op)

    Param.__abstractmethods__ = frozenset()


_complete_ops()