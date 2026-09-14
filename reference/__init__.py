"""Independent architectural reference model for the neural-network accelerator."""

from .functional import (
    AcceleratorConfig,
    Config,
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
    ResultReadout,
    SampleContext,
)

__all__ = [
    "AcceleratorConfig",
    "Config",
    "FunctionalReference",
    "ReferenceConfig",
    "Sample",
    "SampleRecord",
    "CycleConfig",
    "CycleInputs",
    "CycleReference",
    "CycleSnapshot",
    "ResultReadout",
    "SampleContext",
]
