"""Punto de entrada de la API; carga backend/.env antes de crear la aplicación."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)

    import uvicorn
    from fakenews_es.api.main import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
