from dataclasses import dataclass

from .qsystem_base import *


@dataclass(kw_only=True)
class SimpleSystem(QSystem):
    N: int