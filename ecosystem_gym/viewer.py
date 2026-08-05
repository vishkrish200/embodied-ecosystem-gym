"""Thin live and replay viewer adapters over :class:`EcosystemEnv`.

The viewer deliberately caches only values returned by Gym.  It never derives
world state or applies an action itself: reset, step, and render are always
forwarded to the environment instance it wraps.
"""

from __future__ import annotations

import json
import ipaddress
import secrets
import struct
import zlib
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import RLock, Thread
from typing import Any, Mapping

import numpy as np

from .env import EcosystemEnv
from .config import EcosystemConfig
from .trajectory import LEGACY_SCHEMA_VERSION, ReplayResult, SCHEMA_VERSION, TrajectoryWriter, _json_value


GymAction = dict[str, np.ndarray | int]
GymReset = tuple[dict[str, Any], dict[str, Any]]
GymStep = tuple[dict[str, Any], float, bool, bool, dict[str, Any]]
MAX_JSON_BYTES = 16 * 1024


def _as_gym_action(action: Mapping[str, Any]) -> GymAction:
    """Convert a JSON-compatible skill action to the public Gym action shape."""

    return {
        "kind": int(action["kind"]),
        "target": np.asarray(action["target"], dtype=np.float32),
        "duration": np.asarray(action["duration"], dtype=np.float32),
    }


def _replay_records(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError("trajectory has no records")
    version = records[0].get("schema_version")
    if version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION} or any(record.get("schema_version") != version for record in records):
        raise ValueError("trajectory schema version is unsupported")
    header = records[0] if records[0].get("record_type") == "episode_metadata" else None
    steps = records[1:] if header is not None else records
    if not steps:
        raise ValueError("trajectory has no step records")
    seed = (header or steps[0]).get("seed")
    if not isinstance(seed, int) or any(record.get("seed") != seed for record in steps):
        raise ValueError("trajectory must contain one integer seed")
    return steps, header


@dataclass(frozen=True, slots=True)
class ViewerSnapshot:
    """The last observable Gym result, suitable for a small UI or integration."""

    step: int
    observation: dict[str, Any] | None
    reset_info: dict[str, Any] | None
    info: dict[str, Any] | None
    reward: float | None
    terminated: bool
    truncated: bool
    frame: np.ndarray | None


class ViewerSession:
    """Drive one live episode or replay through an authoritative environment.

    ``trace_path`` is intentionally episode-scoped.  A call to ``reset`` starts
    a fresh JSONL file there, so callers must provide a seed whenever recording
    to retain a replayable trajectory.
    """

    def __init__(
        self,
        env: EcosystemEnv | None = None,
        *,
        trace_path: str | Path | None = None,
        episode_id: str = "viewer",
    ) -> None:
        self.env = env or EcosystemEnv(render_mode="rgb_array")
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._episode_id = episode_id
        self._writer: TrajectoryWriter | None = None
        self._step = 0
        self._seed: int | None = None
        self._observation: dict[str, Any] | None = None
        self._reset_info: dict[str, Any] | None = None
        self._info: dict[str, Any] | None = None
        self._reward: float | None = None
        self._terminated = False
        self._truncated = False
        self._frame: np.ndarray | None = None
        self._lock = RLock()

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> GymReset:
        """Reset the wrapped Gym environment and, if requested, open its trace."""

        with self._lock:
            if self._trace_path is not None and seed is None:
                raise ValueError("a recorded viewer session requires an explicit reset seed")
            observation, info = self.env.reset(seed=seed, options=options)
            self._step = 0
            self._seed = seed
            self._observation = observation
            self._reset_info = info
            self._info = None
            self._reward = None
            self._terminated = False
            self._truncated = False
            self._frame = self.env.render()
            if self._trace_path is not None:
                self._replace_writer(
                    TrajectoryWriter(
                        self._trace_path,
                        episode_id=self._episode_id,
                        seed=seed,
                        reset_options=options or {},
                        observation_mode=self.env.config.observation_mode,
                        config=self.env.config,
                    )
                )
            return observation, info

    def step(self, action: GymAction, *, record: bool = True) -> GymStep:
        """Apply a public skill action through ``EcosystemEnv.step`` unchanged."""

        with self._lock:
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._step += 1
            self._observation = observation
            self._info = info
            self._reward = reward
            self._terminated = terminated
            self._truncated = truncated
            self._frame = self.env.render()
            if record and self._writer is not None:
                self._writer.record(
                    step=self._step,
                    action=action,
                    observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return the current environment frame without synthesizing one in the viewer."""

        with self._lock:
            self._frame = self.env.render()
            return self._frame

    def snapshot(self) -> ViewerSnapshot:
        with self._lock:
            return ViewerSnapshot(
                step=self._step,
                observation=self._observation,
                reset_info=self._reset_info,
                info=self._info,
                reward=self._reward,
                terminated=self._terminated,
                truncated=self._truncated,
                frame=self._frame,
            )

    def replay(self, path: str | Path) -> ReplayResult:
        """Replay and validate a recorded trace through this session's Gym calls."""

        records, header = _replay_records(path)
        seed = int(records[0]["seed"])
        options = dict(header.get("reset_options", {})) if header is not None else None
        config_values = dict(header.get("config", {})) if header is not None else {}
        observation_mode = str(config_values.get("observation_mode", (header or records[0]).get("observation_mode", "state_oracle")))
        config_values["observation_mode"] = observation_mode
        replay_config = EcosystemConfig(**config_values)
        with self._lock:
            self._close_writer()
            if self.env.config != replay_config:
                self.env.close()
                self.env = EcosystemEnv(replay_config, render_mode="rgb_array")
            trace_path, self._trace_path = self._trace_path, None
            try:
                self.reset(seed=seed, options=options)
            finally:
                self._trace_path = trace_path
            for expected in records:
                action = _as_gym_action(expected["action"])
                observation, reward, terminated, truncated, info = self.step(action, record=False)
                step = expected["step"]
                if self._step != step:
                    raise ValueError(f"step mismatch at step {step}")
                if info["outcome"] != expected["outcome"]:
                    raise ValueError(f"outcome mismatch at step {step}")
                if not np.isclose(reward, expected["reward"]):
                    raise ValueError(f"reward mismatch at step {step}")
                if bool(terminated) != expected["terminated"] or bool(truncated) != expected["truncated"]:
                    raise ValueError(f"terminal state mismatch at step {step}")
                for key in ("disturbance", "post_disturbance_completion", "camera_sector"):
                    if key in expected and info.get(key) != expected[key]:
                        raise ValueError(f"{key} mismatch at step {step}")
                if _json_value(observation) != expected["observation"]:
                    raise ValueError(f"observation mismatch at step {step}")
            return ReplayResult(steps=len(records), task_success=bool(records[-1]["task_success"]))

    def close(self) -> None:
        with self._lock:
            self._close_writer()
            self.env.close()

    def __enter__(self) -> "ViewerSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _replace_writer(self, writer: TrajectoryWriter) -> None:
        self._close_writer()
        self._writer = writer

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _ui_value(value: Any) -> Any:
    """Keep the stdlib UI small while retaining scalar state exactly."""

    if isinstance(value, np.ndarray) and value.size > 64:
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _ui_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ui_value(item) for item in value]
    return _json_value(value)


class LocalViewerServer:
    """A dependency-free local control panel for a :class:`ViewerSession`.

    The server deliberately has no simulation model.  Its API calls the session,
    which in turn calls Gym, so browser actions and programmatic actions use the
    identical execution and trace path.
    """

    def __init__(self, session: ViewerSession, host: str = "127.0.0.1", port: int = 0) -> None:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError("LocalViewerServer only supports loopback hosts")
        self.session = session
        self._control_token = secrets.token_urlsafe(32)
        handler = self._handler_type()
        # MuJoCo renderers own thread-affine graphics resources, so requests
        # must remain serialized on the one server thread that owns the Gym.
        self._server = HTTPServer((host, port), handler)
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> str:
        if self._thread is None:
            self._thread = Thread(target=self._server.serve_forever, name="ecosystem-viewer", daemon=True)
            self._thread.start()
        return self.url

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
            self._thread = None
        self._server.server_close()

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        session = self.session
        control_token = self._control_token
        html = _HTML.replace("__CONTROL_TOKEN__", control_token)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send_html(html)
                elif path == "/api/state":
                    self._send_json(_snapshot_payload(session))
                elif path == "/api/frame":
                    frame = session.render()
                    if frame is None:
                        self.send_error(HTTPStatus.CONFLICT, "reset the viewer before requesting a frame")
                    else:
                        self._send_png(frame)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                submitted_token = self.headers.get("X-Viewer-Token", "")
                if not secrets.compare_digest(submitted_token, control_token):
                    self.send_error(HTTPStatus.FORBIDDEN, "missing or invalid viewer control token")
                    return
                try:
                    payload = self._read_json()
                    if self.path == "/api/reset":
                        seed = payload.get("seed")
                        if seed is not None:
                            seed = int(seed)
                        session.reset(seed=seed, options=payload.get("options"))
                    elif self.path == "/api/step":
                        session.step(_as_gym_action(payload))
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                except (KeyError, TypeError, ValueError) as error:
                    self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(_snapshot_payload(session))

            def _read_json(self) -> dict[str, Any]:
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type != "application/json":
                    raise ValueError("Content-Type must be application/json")
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ValueError("missing Content-Length")
                length = int(raw_length)
                if length < 0 or length > MAX_JSON_BYTES:
                    raise ValueError("request body too large")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

            def _send_html(self, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
                encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_png(self, frame: np.ndarray) -> None:
                encoded = _png_bytes(frame)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        return Handler


def _snapshot_payload(session: ViewerSession) -> dict[str, Any]:
    snapshot = session.snapshot()
    return {
        "step": snapshot.step,
        "observation": _ui_value(snapshot.observation),
        "reset_info": _ui_value(snapshot.reset_info),
        "info": _ui_value(snapshot.info),
        "reward": snapshot.reward,
        "terminated": snapshot.terminated,
        "truncated": snapshot.truncated,
        "frame_shape": list(snapshot.frame.shape) if snapshot.frame is not None else None,
    }


_HTML = """<!doctype html>
<meta charset="utf-8">
<title>Embodied Ecosystem viewer</title>
<style>body{font:14px system-ui;margin:2rem;max-width:50rem}button,input{font:inherit;margin-right:.4rem}figure{margin:1rem 0}img{display:block;background:#111;border:1px solid #555}pre{background:#f3f3f3;padding:1rem;overflow:auto}</style>
<h1>Embodied Ecosystem viewer</h1>
<p>This is the live MuJoCo room. It starts on seed 7; every control calls the wrapped Gym environment.</p>
<input id="seed" type="number" value="7" aria-label="seed"><button onclick="reset()">Reset</button>
<p><label>x <input id="x" type="number" value="0" step="0.1"></label><label>y <input id="y" type="number" value="0" step="0.1"></label><button onclick="walk()">Walk to</button></p>
<p><button onclick="step(5)">Idle</button><button onclick="step(1)">Pick up</button><button onclick="step(2)">Consume</button></p>
<figure><figcaption>Current room frame</figcaption><img id="frame" width="320" height="240" alt="MuJoCo room frame"></figure>
<pre id="state">Loading…</pre>
<script>
function refreshFrame() { document.querySelector('#frame').src = '/api/frame?' + Date.now(); }
const controlToken = '__CONTROL_TOKEN__';
async function request(path, body) { const r = await fetch(path, body && {method:'POST',headers:{'Content-Type':'application/json','X-Viewer-Token':controlToken},body:JSON.stringify(body)}); const payload = await r.json(); document.querySelector('#state').textContent = JSON.stringify(payload, null, 2); if (!r.ok) throw new Error(payload.error || r.statusText); refreshFrame(); }
async function reset() { await request('/api/reset', {seed:Number(document.querySelector('#seed').value)}); }
function step(kind) { request('/api/step', {kind, target:[0,0], duration:0.1}); }
function walk() { request('/api/step', {kind:0, target:[Number(document.querySelector('#x').value), Number(document.querySelector('#y').value)], duration:5}); }
reset();
</script>"""


def _png_bytes(frame: np.ndarray) -> bytes:
    """Encode the authoritative RGB render without adding a viewer dependency."""

    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("viewer frame must be an HxWx3 uint8 RGB image")

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(
        b"IDAT", zlib.compress(scanlines)
    ) + chunk(b"IEND", b"")
