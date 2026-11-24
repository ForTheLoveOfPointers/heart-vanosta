"""Abstract simulator interface.

``CircAdaptSimulator`` implements this contract.  The interface is deliberately
small so that the CircAdapt adapter can be swapped or wrapped without touching
the sensor, dataset, or model code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Beat, PatientParams, Setting


class Simulator(ABC):
    @abstractmethod
    def simulate(self, patient: PatientParams, setting: Setting) -> Beat:
        """Return a single beat of physiology and its ground-truth labels."""
        raise NotImplementedError
