from dataclasses import dataclass
from collections.abc import Sequence

import dynamiqs as dq

from .qsystem_base import *
from .simple_system import *


@dataclass
class CombinedSystem(QSystem):
    subsystems: Sequence[SimpleSystem]
    
    
    @property
    def Ns(self):
        return [subsys.N for subsys in self.subsystems]
    
    
    def subsys_ops_sum(self, ops):
        
        def eval_term(i, op):
            cmp = [dq.eye(subsys.N) for subsys in self.subsystems]
            cmp[i] = op
            return dq.tensor(*cmp)
        
        for i, op in enumerate(ops):
            term = eval_term(i, op)
            if i == 0:
                res = term
            else:
                res += term
                
        return res
        
    
    def Hamiltonian(self):
        Hs = [subsys.Hamiltonian() for subsys in self.subsystems]
        return self.subsys_ops_sum(Hs)