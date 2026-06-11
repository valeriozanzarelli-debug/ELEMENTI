#!/usr/bin/env python3
"""Server web super motore Lotto — quintina in, ambo/terno/quaterna out."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from super_engine_api import ALL_WHEELS, predict_from_quintina

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path.endswith(".js"):
            js = WEB_DIR / Path(path).name
            if js.is_file():
                return self._send(200, js.read_bytes(), "application/javascript; charset=utf-8")
        if path == "/api/wheels":
            return self._send(200, json.dumps({"wheels": ALL_WHEELS}).encode())
        if path == "/health":
            return self._send(200, b'{"ok":true}')
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            return self._send(404, b'{"error":"not found"}')
        try:
            data = self._read_json()
            quintina = [int(x) for x in data["quintina"]]
            result = predict_from_quintina(data.get("wheel", "NAZIONALE"), quintina)
            self._send(200, json.dumps(result, ensure_ascii=False).encode())
        except (KeyError, TypeError, ValueError) as e:
            self._send(400, json.dumps({"error": str(e)}, ensure_ascii=False).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}, ensure_ascii=False).encode())


def main() -> int:
    if not WEB_DIR.is_dir():
        print(f"Cartella web mancante: {WEB_DIR}")
        return 1
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Super Motore Lotto → http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
