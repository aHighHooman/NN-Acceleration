"""Independent architectural reference model for the neural-network accelerator."""

from .functional import (
    FunctionalReference,
    ReferenceConfig,
    Sample,
    SampleRecord,
)
from .cycle import (
    CycleConfig,
    CycleInputs,
    CycleReference,
    CycleSnapshot,
    ResultEntry,
    SampleContext,
    SampleTiming,
)

__all__ = [
    "FunctionalReference",
    "ReferenceConfig",
    "Sample",
    "SampleRecord",
    "CycleConfig",
    "CycleInputs",
    "CycleReference",
    "CycleSnapshot",
    "ResultEntry",
    "SampleContext",
    "SampleTiming",
]
