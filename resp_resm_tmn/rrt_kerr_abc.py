from collections.abc import Callable

from .rrt_abc import *
from ..transmon import *


class RespResmTmnKerrABC(RespResmTmnABC):
    """:class:`~dq_systems.resp_resm_tmn.rrt_abc.RespResmTmnABC` with the transmon fixed to a Kerr oscillator.

    Convenience layer: builds the ``tmn`` subsystem as a
    :class:`~dq_systems.transmon.tmn_kerr.TransmonKerr` from
    ``(N_tmn, Ec, omega_qub)`` directly, so subclasses/callers don't need to
    construct the transmon themselves. Still abstract w.r.t.
    ``V_interact``/``relax_ops`` (inherited from
    :class:`~dq_systems.resp_resm_tmn.rrt_abc.RespResmTmnABC`).
    """

    def __init__(self,
                 N_res: int,
                 omega_res: float | Callable,
                 J: float,
                 N_tmn: int,
                 Ec: float,
                 omega_qub: float | Callable
                ):

        tmnk = TransmonKerr(
            N=N_tmn,
            Ec=Ec,
            omega_qub=omega_qub
        )
        super().__init__(N_res, omega_res, J, tmnk)