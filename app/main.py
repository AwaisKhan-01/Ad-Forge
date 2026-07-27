"""AdForge Studio v2 entrypoint — serves the FastAPI backend + web UI.

Run:  python -m app.main   (or: uvicorn app.api:app --host 0.0.0.0 --port 7860)
"""
import uvicorn

from app.api import app  # noqa: F401  (re-export for `uvicorn app.main:app`)

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=7860)
