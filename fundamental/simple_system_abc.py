from dataclasses import dataclass

from .qsystem_abc import *


@dataclass
class SimpleSystemABC(QSystemABC):
    N: int