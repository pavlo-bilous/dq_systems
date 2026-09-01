# dq_systems

Interfacing layer between upstream physical modeling (e.g. an FEM simulation
reduced to a lumped-element circuit) and [`dynamiqs`](https://github.com/dynamiqs/dynamiqs),
used to simulate the resulting quantum dynamics. Physical components expose
constructor arguments as "ports" -- lumped-element parameters, either static
numbers or time-dependent callables -- and each component knows how to turn
its own parameters into a `dynamiqs` Hamiltonian.

## Architecture

![dq_systems class hierarchy](docs/class_hierarchy_dark.svg#gh-dark-mode-only)
![dq_systems class hierarchy](docs/class_hierarchy_light.svg#gh-light-mode-only)

Solid grey lines with a hollow arrowhead are inheritance (*is-a*); dashed
copper lines with a diamond are composition (*has-a*) -- a field that holds
an instance of the pointed-to class. A dashed, accent-colored card border
marks an abstract class that Python refuses to instantiate directly until
every `@abstractmethod` is filled in. The copper dot marks the four classes
that actually declare a `Param` "port" (see below).

Equivalent as plain text:

```
QSystemABC (abstract: .Hamiltonian())
├── SimpleSystemABC        -- one "atomic" mode, Hilbert dim N
│   ├── ResonatorMode       -- H = omega * n
│   └── TransmonABC         -- parametrized by (Ec, Ej)
│       └── TransmonKerr    -- Ec/Ej derived from (Ec, omega_qub); Kerr-oscillator H
└── CombinedSystem          -- tensor-product composition of several SimpleSystemABC
    └── resp_resm_tmn/*     -- resonator split into +/- normal modes, coupled to a transmon
        ├── RespResmTmnABC            (abstract: .V_interact(), .relax_ops())
        ├── RespResmTmnKerrABC        (+ transmon = TransmonKerr)
        ├── RespResmTmnKerrDirect     (+ direct coupling, Purcell-filter relaxation)
        └── RespResmTmnKerrDirectDriven (+ external drives, rotating frame)
```

**`Param`** (`fundamental/param_class.py`) is the "port" type: it wraps a
parameter that may be a plain number (static) or a `Callable[[float],
complex]` (time-dependent). A static `Param` behaves like an ordinary
complex number everywhere (arithmetic delegated to the wrapped value). A
time-dependent `Param`, when multiplied by a `dynamiqs` operator (always
`Param * operator`, not the other way around -- see Known issues), produces
a `dynamiqs` time-modulated array. This is the mechanism that lets the same
Hamiltonian-construction code handle both static and driven parameters
uniformly.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

`conftest.py` (repo root) puts the repo's parent directory on `sys.path` so
`import dq_systems` resolves without installing the package -- the package
name *is* this repository's directory name. Tests live under `tests/`,
mirroring the package layout.

## Known issues

Found while writing tests; not fixed here (documentation/tests-only pass).

- **`transmon/tmn_kerr.py`, `TransmonKerr.eval_omega_qub`**: uses
  `jnp.sqrt(...)`, but `jnp` (`jax.numpy`) is never imported into that
  module's namespace, so calling this staticmethod directly raises
  `NameError`. Not hit internally -- `TransmonKerr.__init__` only ever calls
  `eval_Ej`, the other direction. Reproduced by the (`xfail`) test
  `tests/test_transmon.py::TestTransmonKerrEnergyConversion::test_eval_omega_qub_and_eval_Ej_are_inverses`.
  Fix: add `import jax.numpy as jnp`.

- **`fundamental/param_class.py`, `Param.__eq__`**: `Param` is a
  `@dataclass(frozen=True)` subclassing `numbers.Complex`; the dataclass
  decorator generates a concrete `__eq__` (field-tuple equality) before the
  module's metaprogramming step runs, so `__eq__` never gets the
  delegate-to-wrapped-value treatment the other operators get. Practical
  effect: `Param(2.0) == Param(2.0)` is `True`, but `Param(2.0) == 2.0` is
  `False` -- unlike every other arithmetic operator, which does behave like
  the plain wrapped number. See `tests/test_param.py::TestStaticParam::test_equality_is_dataclass_field_equality_not_delegated`.

- **`fundamental/param_class.py`, `Param.__rmul__` vs. dynamiqs operators**:
  `Param.__mul__(self, other)` correctly special-cases a time-dependent
  `Param` times a `dynamiqs` operator (`Param(...) * dq.number(N)` works).
  The reverse order, `dq.number(N) * Param(...)`, does *not* fall back to
  `Param.__rmul__` as one might expect from Python's operator protocol,
  because `dynamiqs.QArray.__mul__` raises `NotImplementedError` for an
  unrecognized right-hand operand instead of returning `NotImplemented`,
  which suppresses Python's normal reflected-operator fallback. Not an
  issue in practice: every call site in this codebase always writes
  `param * operator`. See
  `tests/test_param.py::TestTimeDependentParam::test_rmul_does_not_actually_work_with_a_dynamiqs_operand`.
