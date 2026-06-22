from dataclasses import dataclass

from ....general import *


@dataclass(kw_only=True)
class Transmon(SimpleSystem):
    Ec: float
    Ej: float