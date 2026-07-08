from fastapi import FastAPI

""" What's happening here?
* `FastAPI()` creates the application instance. The `title`, `description`, and `version` parameters are optional metadata for the API documentation.
* `@app.get("/health")` is a decorator that defines a GET endpoint at the path `/health`. When this endpoint is accessed, the `health_check` function is called.
"""

""" Containerization with Docker:
What each line does in `Dockerfile`:
* `FROM python:3.13-slim` — base image. Slim means a minimal Linux with Python 3.13 pre-installed, nothing extra
* `WORKDIR /app` — sets the working directory inside the container. All subsequent commands run from here
* `COPY requirements.txt .` — copies just the requirements file first (reason: Docker builds in layers, and this allows caching of dependencies if `requirements.txt` hasn't changed)
* `RUN pip install` — installs dependencies. `--no-cache-dir` keeps the image size smaller
* `COPY . .` — copies the rest of the backend code into the container
* `CMD [...]` — the command that runs when the container starts. `app.main:app` means "find the `app` object inside `app/main.py`"
"""

""" Docker Compose (For Dummies):
What each line does:
* `services:` — defines the containers that make up the system. 
* `build: ./backend` — tells Compose to build the image using the Dockerfile inside `./backend`
* `ports: "8000:8000"` — maps port 8000 on your Mac to port 8000 inside the container. Format is `host:container`
* `volumes: ./backend:/app` — this is the key one. It mounts your local `backend/` folder into `/app` inside the container. This means code changes reflect immediately without rebuilding.
* `env_file: .env` — passes environment variables from your `.env` file into the container. Empty for now, but we'll need it later for API keys
"""
from app.routes import router

app = FastAPI(
    title="AutoML Platform",
    description="Agentic AutoML backend",
    version="0.1.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}

app.include_router(router) 