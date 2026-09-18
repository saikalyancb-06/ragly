"""Run the backend:  python -m ragly_backend"""
import logging

import uvicorn

from .config import settings
from .offline import is_loopback_host


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not is_loopback_host(settings.host):
        raise SystemExit("RAGLY_HOST must be a loopback address (127.0.0.1) - Ragly never listens on the network.")
    uvicorn.run("ragly_backend.api:api", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
