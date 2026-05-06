"""HTTP policy server for BRX pi05 checkpoints.

Run this from the openpi environment, not from Isaac Lab.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from PIL import Image

from openpi.policies import policy_config
from openpi.training import config as _config


parser = argparse.ArgumentParser(description="BRX openpi HTTP policy server.")
parser.add_argument("--config-name", type=str, default="pi05_brx_finetune")
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8777)
parser.add_argument("--default-prompt", type=str, default="grab the small blocks pick them up and put them in the bucket")
args = parser.parse_args()


def _decode_png_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value.encode("ascii"))
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _make_example(payload: dict[str, Any]) -> dict[str, Any]:
    images_payload = payload["images"]
    images = {
        "left_eye": _decode_png_b64(images_payload["left_eye"]),
        "left_wrist": _decode_png_b64(images_payload["left_wrist"]),
        "right_wrist": _decode_png_b64(images_payload["right_wrist"]),
    }
    images["right_eye"] = _decode_png_b64(images_payload["right_eye"]) if "right_eye" in images_payload else images["left_eye"]
    return {
        "state": np.asarray(payload["state"], dtype=np.float32),
        "images": images,
        "prompt": payload.get("prompt") or args.default_prompt,
    }


print(f"[BRX policy server] loading config={args.config_name}")
train_config = _config.get_config(args.config_name)
print(f"[BRX policy server] loading checkpoint={args.checkpoint_dir}")
policy = policy_config.create_trained_policy(train_config, str(args.checkpoint_dir))
print("[BRX policy server] ready")


class Handler(BaseHTTPRequestHandler):
    server_version = "BRXPolicyHTTP/0.1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True})
            return
        self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json(404, {"ok": False, "error": f"unknown path: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = policy.infer(_make_example(payload))
            actions = np.asarray(result["actions"], dtype=np.float32)
            self._send_json(
                200,
                {
                    "ok": True,
                    "actions": actions.tolist(),
                    "shape": list(actions.shape),
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": repr(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[BRX policy server] {self.address_string()} - {fmt % args}")


server = ThreadingHTTPServer((args.host, args.port), Handler)
print(f"[BRX policy server] listening on http://{args.host}:{args.port}")
server.serve_forever()
