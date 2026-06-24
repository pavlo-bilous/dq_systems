import jax as _jax
_jax.config.update("jax_enable_x64", True)

if not _jax.config.jax_enable_x64:
    raise RuntimeError("JAX is required to be in x64 mode.")