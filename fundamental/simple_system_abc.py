from dataclasses import dataclass

from .qsystem_abc import *


@dataclass
class SimpleSystem(QSystem):
    N: int