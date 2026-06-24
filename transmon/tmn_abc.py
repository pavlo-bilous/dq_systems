from dataclasses import dataclass

from ..fundamental import *


@dataclass
class TransmonABC(SimpleSystemABC):
    Ec: float
    Ej: float