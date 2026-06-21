"""Public, process-independent integration boundary for Oil Gestures ML."""

from oil_gestures.integration.client import iter_events
from oil_gestures.integration.contracts import (
    CAMERA_FRAME_CONTRACT,
    CONTRACT_VERSION,
    RUNTIME_CONTRACT,
    CameraFrameEvent,
    MLRuntimeEvent,
)
from oil_gestures.integration.publisher import (
    MLIntegrationPublisher,
    MLIntegrationPublisherConfig,
)

__all__ = [
    "CAMERA_FRAME_CONTRACT",
    "CONTRACT_VERSION",
    "RUNTIME_CONTRACT",
    "CameraFrameEvent",
    "MLIntegrationPublisher",
    "MLIntegrationPublisherConfig",
    "MLRuntimeEvent",
    "iter_events",
]
