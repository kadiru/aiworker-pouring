"""LAN preview and commands; all simulator mutations stay on the main thread."""
import ctypes
import io
import json
import queue
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread


class Preview:
    def __init__(self, state, directory="/workspace/viewer", allowed_commands=None):
        self.state = state
        self.commands = queue.Queue(maxsize=32)
        self.jpeg = b""
        self.cameras = {}
        self.pending = False
        self.last_capture = 0
        self.pointer = ctypes.pythonapi.PyCapsule_GetPointer
        self.pointer.restype = ctypes.c_void_p
        self.pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        preview = self
        allowed_commands = allowed_commands or {"start", "save", "discard", "reset", "pause", "camera", "cup_view", "water_view", "motion", "probe", "start_xr", "comparison_view", "spill_view", "surface_legacy", "surface_thin", "spill_1", "spill_2", "spill_3", "spill_4", "spill_5"}

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                camera_name = path.removeprefix("/camera/").removesuffix(".jpg")
                if path not in ("/health", "/snapshot.jpg") and not path.startswith("/camera/"):
                    return super().do_GET()
                body = json.dumps(preview.state).encode() if path == "/health" else (preview.cameras.get(camera_name,b"") if path.startswith("/camera/") else preview.jpeg)
                if not body:
                    return self.send_error(503)
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if path == "/health" else "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if self.path != "/control" or self.headers.get("Origin") not in (None, "http://192.168.68.60:8212"):
                    return self.send_error(403)
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("size")
                    command = json.loads(self.rfile.read(length))["command"]
                    if command not in allowed_commands:
                        raise ValueError("command")
                    preview.commands.put_nowait(command)
                except (ValueError, KeyError, queue.Full):
                    return self.send_error(400)
                self.send_response(202)
                self.end_headers()

        self.server = ThreadingHTTPServer(("192.168.68.60", 8212), Handler)
        Thread(target=self.server.serve_forever, daemon=True).start()

    def capture(self):
        if self.pending or time.monotonic() - self.last_capture < 0.2:
            return
        from omni.kit.viewport.utility import capture_viewport_to_buffer, get_active_viewport
        self.pending = True
        self.last_capture = time.monotonic()
        self.handle = capture_viewport_to_buffer(get_active_viewport(), self.on_buffer)

    def on_buffer(self, buffer, size, width, height, byte_format):
        try:
            from PIL import Image
            if size != width * height * 4:
                raise ValueError(f"Unexpected pixel format: {byte_format}")
            raw = ctypes.string_at(self.pointer(buffer, None), size)
            data = io.BytesIO()
            Image.frombytes("RGBA", (width, height), raw).convert("RGB").save(data, "JPEG", quality=82)
            self.jpeg = data.getvalue()
            self.state["preview_updated_unix"] = time.time()
            self.state.pop("preview_error", None)
        except Exception as error:
            self.state["preview_error"] = str(error)
        finally:
            self.pending = False
