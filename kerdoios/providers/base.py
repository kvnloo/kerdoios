from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import ResourceOffer


class ResourceProvider(ABC):
    name: str

    @abstractmethod
    def discover(self) -> list[ResourceOffer]:
        raise NotImplementedError
