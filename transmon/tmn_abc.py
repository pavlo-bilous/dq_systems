from dataclasses import dataclass
from collections.abc import Callable

from ..fundamental import *


@dataclass
class TransmonABC(SimpleSystemABC):
    Ec: float
    Ej: float | Callable
    
    
    def __post_init__(self):
        self.Ej = Param(self.Ej)