# Agentic AutoML Platform
> **(Or, the time I used gradient descent to destroy the gradient descent)**

The Agentic AutoML Platform is an end-to-end machine learning system that automates the full lifecycle of tabular ML workflows using an LLM-driven orchestration layer. Instead of treating AutoML as a black-box model-selection tool, this system decomposes the ML pipeline into modular, composable tools exposed via an MCP (Model Context Protocol) server and orchestrated by a LangGraph-based agent.
 
The system allows users to upload structured datasets (CSV) and interact in natural language to perform data analysis, model training, evaluation, and experimentation. The core idea is to shift AutoML from static optimization into an agentic workflow system with observability, control, and extensibility.

**Key Features**
1. **Natural Language ML Workflow Interface**: Users can describe tasks such as: *“Predict house prices”*, *“Compare XGBoost vs Random Forest”*; the agent translates these into structured ML workflows.
2. **Agentic AutoML Engine (LangGraph-based)**: The system uses a graph-based reasoning engine that dynamically constructs ML pipelines. The agent is a *ReAct-style agent* (Reason + Act loop) following the general pattern:
```
user message
    ↓
LLM thinks: "I should call profile_dataset first"
    ↓
calls profile_dataset via MCP
    ↓
LLM thinks: "Now I should detect the problem type"
    ↓
calls detect_problem_type
    ↓
... continues until it decides it's done
    ↓
returns final answer
 ```

The agent is capable of ingesting data, detecting the type of problem (classification, regression, forecasting), profiling and cleaning data, handling imbalanced classes, performing principal component analysis (PCA), hyperparameter tuning, followed by training and reporting the best model for the task. Unlike traditional AutoML systems, each stage is explicitly visible and controllable.

3. **MCP Tooling Layer**: All ML capabilities are exposed as tools via an MCP server. This enables modular execution and extensibility. The LLM does not execute ML logic directly; it orchestrates tools via MCP calls.
 
4. **FastAPI Backend**: A production-style backend exposes the system via `REST APIs`. The backend handles job scheduling, state management, caching results and tool routing via MCP.
 
5. **Experiment Tracking (`MLflow` Integration)**: Every run is fully tracked via `Mlflow`:
 - model parameters
 - metrics (accuracy, RMSE, F1, etc.)
 - feature sets
 - artifacts (models, plots)
 - training logs

This enables reproducibility and comparison across experiments.

6. **Observability Layer (`LangSmith`)**: The system is instrumented with `LangSmith` to track agent decision paths, tool calls, latency per step, prompt versions, and failure modes. This provides full transparency into the agent’s reasoning process. 
7. **Dockerized Architecture**: The entire system is containerized for reproducibility and deployment.
 
---
## Backend Design (via `FastAPI`)

The backend is a production-style `FastAPI` service that acts as the system's front door. It exposes four `REST` endpoints (`/upload`, `/run`, `/job/{id}`, `/predict`) and is deliberately kept thin — its job is request handling and orchestration routing, not ML logic.

### Why `FastAPI` over `Flask`?
`FastAPI` is async-native, which matters here because the `/run` endpoint makes a long-running `HTTP` call to the orchestrator (which itself runs a multi-step LLM+tool loop). With Flask's synchronous model, that call would block the server thread for the entire duration of the agent run. With `FastAPI`'s `async def` endpoints, the event loop can handle other requests while waiting. For a system where a single workflow can take 60–200 seconds, this isn't a minor detail.
`FastAPI` also auto-generates OpenAPI documentation at `/docs` from `Pydantic` models, which was useful for debugging request shapes during development.

### `Request`/`Response` contracts via `Pydantic`
Every endpoint's input and output is defined as a `Pydantic` model in `models.py`. This enforced a discipline that paid off repeatedly: when the `problem_type` field needed to be threaded from `/run` through to `/predict`, having a typed `RunResponse` meant the compiler (and `FastAPI`'s validator) caught mismatches immediately rather than at runtime. The `PredictRequest` model similarly carries `problem_type` so the predict route can branch correctly between `sklearn`, `FFN`, and forecasting paths without inspecting filenames.

### The `/run → orchestrator` handoff
The `/run` endpoint does one direct MCP call itself — `detect_problem_type` — before delegating to the orchestrator. This is intentional: the problem type needs to be returned in the `RunResponse` so the `Gradio` frontend can immediately update its UI (toggling between a file upload widget and an image output for forecasting). Relying on the agent to surface this in free-form text would have been fragile. The rest of the workflow is handed off to the orchestrator via an `httpx.AsyncClient` `POST` to `http://orchestrator:8002/invoke`.

### Artifact path conventions
One of the harder-won lessons from this project was that path conventions across independently-built services are load-bearing. The backend's `/predict` endpoint reconstructs artifact paths that were originally written by the MCP server's tools — it never gets those paths handed to it directly. Two stems drive this:
* `base_cleaned`: the stem for all model artifacts (e.g. `{job_id}_{filename}_cleaned`)
* `base_raw`: split from base_cleaned for encoder and label map files, because clean_dataset saves those before writing the `_cleaned` suffix

Getting these wrong caused cascading failures that were non-obvious to debug, because each individual path looked plausible. The fix was writing them down explicitly and treating them as a convention, not an implementation detail.

### Backend CURL commands

* Uploading new file:
```
curl -X POST http://localhost:8000/upload -F "file=@irrigation_prediction.csv"
```

* Running algorithm:
```
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "4a09744a-5a9b-4154-93dc-740e8ef51a8d",
    "filename": "irrigation_prediction.csv",
    "target_column": "Irrigation_Need"
  }'
```

* Finding files:
```
docker exec -it automl-platform-backend-1 ls /app/data/ | grep 4a09744a-5a9b-4154-93dc-740e8ef51a8d
```

* Prediction:
```
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "4a09744a-5a9b-4154-93dc-740e8ef51a8d",
    "train_filename": "4a09744a-5a9b-4154-93dc-740e8ef51a8d_irrigation_prediction_cleaned.csv",
    "test_filename": "a6111549-af79-4e6f-92e8-16841d761e2d_test_soil.csv",                         
    "model_type": "random_forest"
  }'

```

---
## Dockerized Architecture 

---
## MCP Tool Server

---
## `LangGraph` Orchestrator and `LangSmith` for Observability

```
curl POST /run (FastAPI, port 8000)
        │
        │ HTTP call
        ▼
Orchestrator service (LangGraph, port 8002)
        │
        │  ReAct loop: LLM decides what to call
        ▼
   Gemini 3.1 Flash-Lite
        │
        │  tool calls via langchain-mcp-adapters
        ▼
MCP Server (port 8001)
  ├── profile_dataset
  ├── detect_problem_type
  ├── train_model
  └── evaluate_model
        │
        ▼
   MLflow (port 5001)
```

---
## `scikit-learn` and `TensorFlow` Implementations

---
## Frontend Design and UI

----
## References
* **Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers**, Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan. (2026). [arXiv:2506.13538v5](https://arxiv.org/abs/2506.13538).
* **Model Context Protocol Explained in 3 Levels of Difficulty**, Bala Priya C. (2026). [(here)](https://machinelearningmastery.com/model-context-protocol-explained-in-3-levels-of-difficulty/)
* **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML**, Patara Trirat, Wonyong Jeong, Sung Ju Hwang. (2024). [arXiv:2410.02958](https://arxiv.org/abs/2410.02958).





