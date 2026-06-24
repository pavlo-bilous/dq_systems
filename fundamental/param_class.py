from collections.abc import Callable
from numbers import Complex

import dynamiqs as dq


class Param(Complex):
    
    def __init__(self, p: complex | Callable[[float], complex]):
        self.p = p
        
        
    def __repr__(self):
        return f"Param({self.p})"

    
    @property
    def time_dependent(self) -> bool:
        return isinstance(self.p, Callable)
    
    
    def __call__(self, t: float):
        if self.time_dependent:
            return self.p(t)
        else:
            return self.p
    
    
    def __mul__(self, other):
        if self.time_dependent:
            if isinstance(other, dq.QArrayLike):
                return dq.modulated(self.p, other)
            else:
                raise TypeError("Time-dependend parameter can only be multiplied with dynamiqs object.")
        else:
            return self.p * other
        
        
    def __rmul__(self, other):
        return self * other
    
    
    
def _delegate_op(self, op_name, *args, **kwargs):
    return getattr(self.p, op_name)(*args, **kwargs)


def _write_op_str(op_name):
    op = (
f"""def {op_name}(self, *args, **kwargs):
    return _delegate_op(self, '{op_name}', *args, **kwargs)"""
    )
    return op
    
    
def _complete_ops():
    ops = {}

    for op_name in Param.__abstractmethods__:
        op_str = _write_op_str(op_name)
        exec(op_str, {'_delegate_op': _delegate_op}, ops)

    for op_name, op in ops.items():
        setattr(Param, op_name, op)

    Param.__abstractmethods__ = frozenset()
    
    
_complete_ops()