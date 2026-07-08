""" Crash course on Pydantic
Pydantic lets you define the shape of data using Python classes. When data comes in (say, a JSON request body), Pydantic automatically validates it and gives you a clean Python object. If something's wrong (missing field, wrong type), it rejects it with a clear error before your code even runs.
Three things Pydantic gives us here:
* Input validation on requests
* Consistent response shapes
* Auto-documentation in `/docs` (FastAPI reads the models and generates the schema there)
"""

""" What do we need:
* `UploadResponse` — returned after a CSV is uploaded (job ID, filename, status)
* `RunRequest` — body for `POST /run` (job ID, target column name)
* `RunResponse` — returned after a run starts (job ID, status)
* `JobStatus` — returned by GET /job/{id} (job ID, status, logs)
"""

from pydantic import BaseModel
from typing import Optional, List

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str

class RunRequest(BaseModel):
    job_id: str
    filename: str
    target_column: str
    problem_type: str

class RunResponse(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    results: Optional[dict] = None
    problem_type: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    logs: List[str]

class ResultsResponse(BaseModel):
    job_id: str
    metrics: Optional[dict] = None
    predictions: Optional[List[float]] = None

class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    job_id: str
    train_filename: str
    test_filename: str
    model_type: str
    problem_type: str