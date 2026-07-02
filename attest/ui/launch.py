"""Launch the desktop app: serve the engine locally and open it in a native window.

Uses pywebview for a real desktop window (Cocoa on macOS, GTK/Qt on Linux). If
pywebview isn't installed, it falls back to opening your default browser — so the
app always runs. PyInstaller bundles this into a downloadable app per OS.
"""

from __future__ import annotations

import secrets
import socket
import threading
import time


def _wait_until_up(host: str, port: int, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.1)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run(host: str = "127.0.0.1", port: int | None = None, open_window: bool = True) -> None:
    import uvicorn

    from .server import create_app

    port = port or _free_port()
    # A fresh secret per run: /api rejects requests without it, so a random
    # webpage you visit can't poke this localhost server (browsers happily send
    # cross-origin POSTs to 127.0.0.1). The page reads it from the URL.
    token = secrets.token_urlsafe(16)
    server = uvicorn.Server(uvicorn.Config(create_app(token=token), host=host, port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_until_up(host, port)
    url = f"http://{host}:{port}/?token={token}"

    if open_window:
        try:
            import webview  # pywebview
            webview.create_window("Attest", url, width=1140, height=780, min_size=(940, 640))
            webview.start()
            return
        except Exception as exc:  # noqa: BLE001 - no GUI backend -> open the browser
            print(f"(native window unavailable: {exc}; opening your browser instead)")

    # Browser mode (explicit --no-window, or pywebview fell through): open it for them.
    import webbrowser
    webbrowser.open(url)
    print(f"Attest is running at {url}  (Ctrl-C to stop)")
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
