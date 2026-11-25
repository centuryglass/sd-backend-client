"""A stand-in ComfyUI `main.py` for `tests/test_managed_comfyui_process.py`: serves `/system_stats` and nothing else.

It parses `--listen` and `--port` as ComfyUI does, and reports its own `argv`. `--crash` exits with code 3 before
serving, and `--ignore-term` ignores SIGTERM so a stop has to kill it.
"""
import argparse
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--listen', default='127.0.0.1')
parser.add_argument('--port', type=int, default=8188)
parser.add_argument('--crash', action='store_true')
parser.add_argument('--ignore-term', action='store_true')
args = parser.parse_args()

print('\x1b[32m[INFO]\x1b[0m Starting stub server', flush=True)
sys.stdout.write('loading 50%\rloading 100%\n')
sys.stdout.flush()
if args.crash:
    print('Stub crash requested', flush=True)
    sys.exit(3)
if args.ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


class Handler(BaseHTTPRequestHandler):
    """Answers `/system_stats` with the fields `SystemStatResponse` requires."""

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        if self.path != '/system_stats':
            self.send_error(404)
            return
        body = json.dumps({
            'system': {'os': 'posix', 'python_version': sys.version, 'embedded_python': False,
                       'comfyui_version': '0.0.0', 'pytorch_version': '0', 'argv': sys.argv,
                       'ram_total': 1, 'ram_free': 1},
            'devices': [],
        }).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *log_args) -> None:  # pylint: disable=redefined-builtin
        pass


# On Windows, `SO_REUSEADDR` lets a second socket bind a port in use. ComfyUI's asyncio server doesn't set it there.
HTTPServer.allow_reuse_address = sys.platform != 'win32'
server = HTTPServer((args.listen, args.port), Handler)
print(f'To see the GUI go to: http://{args.listen}:{args.port}', flush=True)
server.serve_forever()
