"""Live web viewer for VLM policy episodes (scripts/robowits/eval/vlm/policy.py).

Every episode writes <episode_dir>/events.jsonl plus the images the model saw (images/) and video.mp4. This
serves index.html, lists the episodes under --root, and streams an episode's events to the browser over
server-sent events, following the file while the episode is still running (e.g. a Slurm job writing to
shared storage).

    source .venv/bin/activate && python scripts/robowits/eval/vlm/viewer/server.py [--root tmp/gs-gym-evals] [--port 8766]

The server binds to 127.0.0.1; from a laptop, forward the port first: ssh -L 8766:127.0.0.1:8766 <login node>.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import mimetypes
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

INDEX_HTML = Path(__file__).resolve().parent / "index.html"
EVENTS_FILE = "events.jsonl"
POLL_INTERVAL = 0.5
# An episode without an episode_end whose log has not changed for this long is reported as stalled
# (a killed or crashed job); a single model call can take minutes, so this is generous.
STALLED_AFTER = 20 * 60
# Episodes sit at <root>/<set>/<run>/<task>/episode_NNN/ (or one level shallower or deeper).
EPISODE_GLOBS = [f"{'*/' * depth}episode_*/{EVENTS_FILE}" for depth in range(1, 5)]


def _read_events(path: Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line still being written
    return events


def summarize_episode(root: Path, events_path: Path) -> dict:
    """One row of the episode list, from the episode's start and end records (or its progress so far)."""
    episode_dir = events_path.parent
    events = _read_events(events_path)
    start = next((e for e in events if e.get("kind") == "episode_start"), {})
    end = next((e for e in reversed(events) if e.get("kind") == "episode_end"), None)
    calls = [e for e in events if e.get("kind") == "model_call"]
    motions = [e for e in events if e.get("kind") == "motion"]
    modified = events_path.stat().st_mtime
    if end is not None:
        status = (
            "crashed"
            if end.get("crashed")
            else "success"
            if end.get("success")
            else "gave_up"
            if end.get("gave_up")
            else "failed"
        )
    else:
        status = "running" if time.time() - modified < STALLED_AFTER else "stalled"
    costs = [c["cost_usd"] for c in calls if c.get("cost_usd") is not None]
    return {
        "id": episode_dir.relative_to(root).as_posix(),
        "group": episode_dir.parent.parent.relative_to(root).as_posix(),
        "task": start.get("task") or episode_dir.parent.name,
        "task_dir": episode_dir.parent.name,
        "episode": start.get("episode", episode_dir.name),
        "model": start.get("model"),
        "harness": start.get("harness"),
        "status": status,
        "steps": (end or {}).get("steps", motions[-1]["step"] if motions else 0),
        "max_steps": start.get("max_episode_steps"),
        "llm_calls": len(calls),
        "max_reward": (end or {}).get("max_reward", motions[-1]["max_reward"] if motions else None),
        "cost_usd": round(sum(costs), 4) if costs else None,
        "started": start.get("ts"),
        "modified": modified,
    }


def list_episodes(root: Path) -> list[dict]:
    paths = {p for pattern in EPISODE_GLOBS for p in root.glob(pattern)}
    rows = []
    for path in paths:
        try:
            rows.append(summarize_episode(root, path))
        except OSError:
            continue
    return sorted(rows, key=lambda r: (r["group"], r["task_dir"], str(r["episode"])))


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's signature
        pass

    def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json", {"Cache-Control": "no-store"})

    def _episode_dir(self, query) -> Path | None:
        """The requested episode directory, if it is a real episode inside --root."""
        rel = (query.get("ep") or [""])[0]
        path = (self.root / rel).resolve()
        if rel and path.is_relative_to(self.root) and (path / EVENTS_FILE).is_file():
            return path
        return None

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            elif url.path == "/api/episodes":
                self._json({"root": str(self.root), "episodes": list_episodes(self.root)})
            elif url.path == "/api/events":
                episode_dir = self._episode_dir(query)
                if episode_dir is None:
                    return self._json({"error": "unknown episode"}, 404)
                self._stream_events(episode_dir, int((query.get("after") or ["0"])[0]))
            elif url.path == "/api/file":
                self._serve_file(query)
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, payload: dict, event: str | None = None) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str)
        self.wfile.write(((f"event: {event}\n" if event else "") + f"data: {data}\n\n").encode())
        self.wfile.flush()

    def _stream_events(self, episode_dir: Path, after: int) -> None:
        """Send the episode's events after `after`, then keep following the file until episode_end."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        path = episode_dir / EVENTS_FILE
        offset, buf, sent_snapshot, last_beat = 0, b"", False, time.time()
        while True:
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            offset += len(chunk)
            buf += chunk
            *lines, buf = buf.split(b"\n")
            ended = False
            for line in lines:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("seq", 0) > after:
                    self._sse(ev)
                ended = ended or ev.get("kind") == "episode_end"
            if not sent_snapshot:
                self._sse({}, event="snapshot_end")
                sent_snapshot = True
            if ended:
                self._sse({}, event="episode_end")
                return
            if time.time() - last_beat > 10:
                self._sse({"modified": path.stat().st_mtime}, event="heartbeat")
                last_beat = time.time()
            time.sleep(POLL_INTERVAL)

    def _serve_file(self, query) -> None:
        """A file inside an episode directory (an image the model saw, or video.mp4), with Range support."""
        episode_dir = self._episode_dir(query)
        rel = (query.get("path") or [""])[0]
        path = (episode_dir / rel).resolve() if episode_dir is not None and rel else None
        if path is None or not path.is_relative_to(episode_dir) or not path.is_file():
            return self._json({"error": "file not found"}, 404)
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        partial = bool(m and (m.group(1) or m.group(2)))
        if partial:
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if partial:  # browsers reject a 206 without Content-Range
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the live VLM policy episode viewer.")
    parser.add_argument("--root", default="tmp/gs-gym-evals", help="Directory searched for episode folders")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.root = Path(args.root).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"VLM episode viewer: http://127.0.0.1:{args.port}/  (episodes under {Handler.root})", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
