"""
Entrypoint module for the local REST inference server.
"""

import argparse
import uvicorn

from edge_vision.api.app import create_app

app = create_app()


def run_server() -> None:
    """CLI runner to launch Uvicorn server with configurable host and port."""
    parser = argparse.ArgumentParser(
        description="Edge Vision Reliability Lab — Inference REST Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (defaults to localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    args = parser.parse_args()

    uvicorn.run("edge_vision.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    run_server()
