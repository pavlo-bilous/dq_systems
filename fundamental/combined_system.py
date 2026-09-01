from dataclasses import dataclass
from collections.abc import Sequence

import dynamiqs as dq

from .qsystem_abc import *
from .simple_system_abc import *


@dataclass
class CombinedSystem(QSystemABC):
    """A composite quantum system built from several independent subsystems.

    Assembles the joint Hilbert space as the tensor product of each
    subsystem's own space (in ``subsystems`` order), and its bare Hamiltonian
    as the sum of each subsystem's Hamiltonian embedded into the full space
    via :meth:`subsys_ops_sum` (identity on every other factor). Subclasses
    typically add interaction/coupling terms on top (see e.g.
    ``resp_resm_tmn.rrt_abc.RespResmTmnABC.Hamiltonian``, which adds
    ``V_interact()`` to ``super().Hamiltonian()``).
    """

    subsystems: Sequence[SimpleSystemABC]


    @property
    def Ns(self):
        """Fock-space truncation dimensions of the subsystems, in tensor-product order."""
        return [subsys.N for subsys in self.subsystems]


    def subsys_ops_sum(self, ops):
        """Embed one operator per subsystem into the joint space and sum them.

        ``ops`` must have one entry per subsystem (in the same order as
        ``self.subsystems``); each ``ops[i]`` acts on subsystem ``i``'s space
        and is padded with identities on every other factor before summing,
        i.e. this computes ``sum_i (I ⊗ ... ⊗ ops[i] ⊗ ... ⊗ I)``. Used both
        for the bare Hamiltonian (see :meth:`Hamiltonian`) and, by
        subclasses, for other subsystem-local operators (e.g. photon-number
        operators for a rotating-frame reference Hamiltonian).
        """

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
        """Bare (uncoupled) Hamiltonian: each subsystem's own Hamiltonian, embedded and summed."""
        Hs = [subsys.Hamiltonian() for subsys in self.subsystems]
        return self.subsys_ops_sum(Hs)