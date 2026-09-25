from __future__ import annotations

import os

from app.config import Settings
from app.core.factory import build_service


def main() -> None:
    import uvicorn
    settings = Settings.from_env()
    settings.ensure_dirs()
    build_service(settings)
    host = os.getenv("FIRST_AID_HOST", "127.0.0.1")
    port = int(os.getenv("FIRST_AID_PORT", "8000"))
    uvicorn.run("app.api.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
