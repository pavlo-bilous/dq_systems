from abc import ABC, abstractmethod
import inspect

import jax.numpy as jnp
import dynamiqs as dq


class QSystem(ABC):
        
    @abstractmethod
    def Hamiltonian(self):
        pass
    
    
    @staticmethod
    def diagonalize(H):
        e, v = jnp.linalg.eigh(H.to_jax())
        v = v.T[:, :, jnp.newaxis]
        return e, dq.asqarray(v, dims=H.dims)
    
    
    @classmethod
    def init_from_dict(cls, d: dict):
        sig = inspect.signature(cls.__init__)
        valid_keys = set(sig.parameters) - {"self"}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)