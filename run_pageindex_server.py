#!/usr/bin/env python3
"""Start server — all logs to terminal (stderr) + logs/app.log"""
import os
import socket
import sys

os.environ["PYTHONUNBUFFERED"] = "1"

PORT = int(os.environ.get("PAGEINDEX_PORT", "8000"))


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    port = PORT
    if not port_is_free(port):
        for alt in (8001, 8002, 8003, 8080):
            if alt != port and port_is_free(alt):
                sys.stderr.write(f"Port {port} busy — using {alt} instead.\n")
                sys.stderr.flush()
                port = alt
                break
        else:
            sys.stderr.write(
                f"Ports 8000-8003 and 8080 are busy. Kill listeners:\n"
                f"  netstat -ano | findstr \":800\"\n"
                f"  taskkill /F /PID <pid>\n"
                f"Or: $env:PAGEINDEX_PORT=\"8090\"; python run_pageindex_server.py\n"
            )
            sys.stderr.flush()
            raise SystemExit(1)

    sys.stderr.write(f"Starting PageIndex server on http://127.0.0.1:{port}\n")
    sys.stderr.flush()

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=port,
        log_level="debug",
        access_log=True,
        reload=False,
        log_config=None,
    )
