from dataclasses import dataclass

from ..fundamental import *


@dataclass
class Transmon(SimpleSystem):
    Ec: float
    Ej: float