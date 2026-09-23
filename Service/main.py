"""Start the Muse emotion recognition web service."""

import sys
from pathlib import Path


# This lets the file run directly: python Service/main.py
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

HOST = "127.0.0.1"
PORT = 5001


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as error:
        print(f"Missing Python package: {error.name}")
        print("Install the packages with: pip install -r requirements.txt")
        return

    print(f"Open http://{HOST}:{PORT} in your browser.")
    print(f"FastAPI documentation: http://{HOST}:{PORT}/docs")
    uvicorn.run("Service.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
