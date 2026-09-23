"""Architecture plugins that classify fact-graph nodes onto traits."""

from .api import (
    Architecture,
    ClassifiedGraph,
    DeviceResource,
    EdgeAction,
    NodeTraits,
    ResourceKind,
    allowed_edge_actions,
    can_split_stages,
    is_group_opportunity,
)
from .fake import FAKE, FakeArch
from .hopper import HOPPER, HOPPER_CUDA_TARGET, HOPPER_RESOURCE, HopperArch

__all__ = [
    "Architecture",
    "ClassifiedGraph",
    "DeviceResource",
    "EdgeAction",
    "FAKE",
    "FakeArch",
    "HOPPER",
    "HOPPER_CUDA_TARGET",
    "HOPPER_RESOURCE",
    "HopperArch",
    "NodeTraits",
    "ResourceKind",
    "allowed_edge_actions",
    "can_split_stages",
    "is_group_opportunity",
]
