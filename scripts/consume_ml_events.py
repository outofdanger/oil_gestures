from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oil_gestures.integration import CAMERA_FRAME_CONTRACT, iter_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Consume Oil Gestures ML contracts.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--show-camera-payload", action="store_true")
    args = parser.parse_args()

    for event in iter_events(args.host, args.port):
        if event.get("contract") == CAMERA_FRAME_CONTRACT and not args.show_camera_payload:
            event = dict(event)
            payload = event.pop("data_base64", "")
            event["data_base64_bytes"] = len(payload)
        print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
