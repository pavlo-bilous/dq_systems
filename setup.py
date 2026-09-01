# Despite the filename, this is *not* a distutils/setuptools build script --
# it is a plain module, imported once for its side effect (see
# dq_systems/__init__.py). JAX defaults to 32-bit floats, which is not
# precise enough for the eigen-decompositions and time-dependent evolution
# dynamiqs performs downstream, so x64 mode is forced here, as early as
# possible, before any other dq_systems or dynamiqs code can create arrays.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

# Defensive check: `update` should always succeed, but if some other import
# re-disabled x64 mode after us, fail loudly rather than silently running at
# reduced precision.
if not _jax.config.jax_enable_x64:
    raise RuntimeError("JAX is required to be in x64 mode.")