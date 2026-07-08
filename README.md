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

The system runs as five containers managed by Docker Compose, sharing a single named volume (`ml_data`) mounted at `/app/data/` in both the backend and MCP server containers. This is the mechanism by which the MCP server writes model artifacts that the backend later reads at `/predict time` — there is no artifact transfer over `HTTP`, just a shared filesystem.
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  frontend   │────▶│   backend    │────▶│ orchestrator │
│  (7860)     │     │   (8000)     │     │   (8002)     │
└─────────────┘     └──────────────┘     └──────────────┘
                          │                      │
                          │                      ▼
                    ┌─────┴──────┐     ┌──────────────────┐
                    │   mlflow   │◀────│   mcp-server     │
                    │  (5001)    │     │    (8001)        │
                    └────────────┘     └──────────────────┘
                                              │
                                       ml_data volume
                                       /app/data/
```

### Why a named volume over a bind mount for data?
The `ml_data` volume is shared between `backend` and `mcp-server`. A named Docker volume (rather than a bind mount to a host directory) was chosen because it avoids the file permission issues that arise when containers run as non-root users and write to host-owned directories. The tradeoff is that artifacts are not directly browsable from the host without `docker exec —` acceptable during development.

The `backend` and `mcp_server` source directories are still bind-mounted (`./backend:/app`, `./mcp_server:/app`) so that code changes reflect immediately without a rebuild.

### Service startup ordering
The MCP server needs to be reachable before the orchestrator finishes initializing, because the orchestrator loads all tools from MCP at startup via get_tools(). Docker Compose's depends_on is not sufficient here — it waits for the container to start, not for the `HTTP` server inside it to be ready. The orchestrator handles this with an explicit retry loop (10 attempts, 3-second delay) before failing hard. This was a real failure mode discovered during early runs.

---
## MCP Tool Server

The MCP (Model Context Protocol) server is where all ML logic lives. It is deliberately isolated from both the FastAPI backend and the orchestrator: neither of those services imports ML libraries. All `scikit-learn`, `XGBoost`, `CatBoost`, `TensorFlow`, and `pandas` code runs inside the MCP server container.

### Why MCP?
MCP is a protocol for exposing callable tools to LLMs in a standardized way. Instead of writing custom function-calling schemas, `FastMCP` generates the JSON schema for each tool automatically from Python type hints and docstrings. The LLM sees a clean list of tool names and signatures; it never touches model files or dataframes directly. This decomposition is the core architectural bet of the project: by treating each ML step as a named, inspectable tool call rather than a monolithic AutoML pipeline, every decision the agent makes is logged, auditable, and individually replaceable.

### Tool design philosophy
Each tool does one thing and returns a structured dict. Side effects (writing to `/app/data/`) are explicit and documented in the return value — every tool that saves a file returns the path it wrote to. This made debugging significantly easier: when a downstream tool failed to find a file, the path mismatch was immediately visible in the tool return logged by `LangSmith`.

The tools are stateless at the protocol level — each call is independent. State persists only through the filesystem (saved CSVs, pickled models, `.npz` arrays) and MLflow. This is a deliberate choice: it means the agent can call tools in any order, retry a failed step, or skip steps entirely, without needing to track intermediate objects in memory.

### The `clean_dataset` ordering problem
The order of operations in clean_dataset matters and was not obvious up front. The current order is:
```
column standardisation → datetime fix → numeric coercion → 
drop constants → drop duplicates → null handling → 
target encoding → feature encoding → save
```

Feature encoders are saved before the `_cleaned.csv` is written, using the raw `dataset_id` stem. This means the encoder path uses the original filename prefix, not the `_cleaned` suffix. Getting this wrong in the `/predict` path — using the cleaned stem to look up encoders — caused silent failures where test data arrived at the model unencoded. The fix required auditing every artifact path against the tool that originally wrote it.

### OHE vs. label encoding
An early version of `clean_dataset` used one-hot encoding for categorical features with low cardinality. This was replaced with label encoding globally after testing on a high-cardinality agricultural dataset where OHE caused `Random Forest` accuracy to collapse from ~100% to ~49% by fragmenting the feature space. The tradeoff is that label encoding imposes an arbitrary ordinal relationship on nominal categories — but in practice, tree-based models (which make up most of the model portfolio) are robust to this, and it avoids the dimensionality explosion from OHE on real-world categoricals.

### `XGBoost` + `sklearn` Pipeline routing
`XGBoost` with `sample_weight` cannot be used inside a `sklearn` Pipeline with the standard `fit()` call, because Pipeline's metadata routing for sample_weight fails in some configurations. The fix: fit the `StandardScaler` separately, transform `X_train` explicitly, then call `model.fit(X_train_scaled, y_train, sample_weight=weights)` directly before re-wrapping both in a Pipeline for a consistent `predict`/`predict_proba` interface. This is a known `sklearn`/`XGBoost` compatibility issue — worth knowing about before hitting it in production.

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

### Why a separate orchestrator service?
The orchestrator runs as its own container rather than being embedded in the `FastAPI` backend. This is a deliberate architectural separation: the orchestrator is async-heavy and long-running, while the backend is request-handling and should stay responsive. Separating them also means the orchestrator can be restarted or swapped independently — if you wanted to replace the `LangGraph` agent with a different reasoning loop, the backend wouldn't need to change.

### `create_react_agent` and the prebuilt `ReAct` loop
The orchestrator uses `LangGraph`'s `create_react_agent`, which implements the full ReAct (Reason + Act) loop without requiring a manually constructed graph. The loop is: LLM receives messages → decides whether to call a tool → tool result is appended to message history → LLM reasons again → repeat until LLM issues a final text response. The entire message history (including all intermediate tool calls and results) is the state that flows through the graph. This was kept as a prebuilt rather than a custom graph because the goal was to understand the interface between LLM reasoning and tool execution before building custom graph topologies.

### `MultiServerMCPClient` and tool loading at startup
`MultiServerMCPClient` from `langchain-mcp-adapters` handles the MCP handshake and converts MCP tool schemas into `LangChain`-compatible tool objects. This happens once at container startup via `FastAPI`'s lifespan context, not on each request. The compiled agent and loaded tools are stored as module-level globals and reused across all incoming requests. This matters for latency — the MCP handshake and `get_tools()` call would add ~300ms per request if done inline.

### Prompt engineering for reliability
The message passed to the agent from `/run` is a structured natural language prompt, not a raw user query. It specifies the exact sequence of tool calls expected, conditional branching (e.g. *"if problem_type is forecasting, do X; if classification or regression, do Y"*), and error handling instructions (*"after every tool call, check whether the result contains 'error'"*). This was necessary because the LLM otherwise made inconsistent decisions about tool ordering — sometimes skipping hyperparameter search, sometimes calling evaluate before train. The prompt functions as a soft workflow specification that the LLM interprets, not a hardcoded pipeline.

### `LangSmith`
Every agent run is traced in `LangSmith`, which captures the full message history, each tool call with its inputs and outputs, token counts, and latency per step. This was essential during debugging — when the agent called a tool with a wrong argument or silently skipped a step, the `LangSmith` trace made the exact failure point visible without adding print statements. In a system where the agent's decisions are not deterministic, having a trace per run is the practical alternative to stepping through a debugger.

---
## `scikit-learn` and `TensorFlow` Implementations

The model portfolio covers four `sklearn`-compatible models (Logistic/Linear Regression, Random Forest, XGBoost, CatBoost), a feedforward neural network (FFN), and two sequence models for forecasting (LSTM, Transformer).

### Why this portfolio?
The four `sklearn` models represent the core progression from linear to ensemble to gradient-boosted methods, which is the standard comparison baseline for tabular ML. The FFN adds a neural baseline on tabular data — it rarely wins against well-tuned tree models on tabular tasks, but training it is instructive, and it serves as an architectural contrast. LSTM and Transformer cover the forecasting path and demonstrate that the same MCP tool interface extends naturally to sequence models with different data preparation requirements.

### Separation of hyperparameter search and training
`hyperparameter_search` and `train_model` are separate MCP tools. This means the agent calls `GridSearchCV` explicitly and then passes `best_params` into train_model — it's not hidden inside training. The separation makes the tuning step visible in `LangSmith` traces and means you can retrain with different params without re-running the grid search, or skip tuning entirely for quick experiments.

### Forecasting data pipeline
The forecasting path required a purpose-built data preparation tool (`prepare_forecast_dataset`) because sequence models need a fundamentally different data structure than tabular models. The tool converts a cleaned CSV into sliding windows of shape `(samples, window_size, n_features)`, fits a `StandardScaler` on input features and a `MinMaxScaler` on the target (separately, to allow independent inverse-transformation of predictions), and saves everything to a `.npz` file. The train/test split happens before scaling to avoid leakage — a common mistake when building time series pipelines. Univariate datasets (no covariate columns) fall back gracefully to using the lagged target as the sole input feature.

### Transformer custom layers and serialization
The Transformer model uses custom `Keras` layers (`MultiHeadSelfAttention`, `TransformerBlock`, `PositionalEncoding`, `TransformerEncoder`). Saving and reloading these requires passing a `custom_objects` dict to `tf.keras.models.load_model()`. This is handled in the `/predict` route in `routes.py`. A subtlety: the custom layer classes are defined in both `server.py` and `routes.py` independently, because the `MCP` server and `FastAPI` backend run in separate containers with no shared Python imports. The `get_config()` method on each layer is required for serialization to work correctly — without it, `model.save()` produces a file that cannot be reloaded.

### `_clean_test_df` and test set integrity
Test data at `/predict` time must go through the same preprocessing steps as training data, but not all of them. The `_clean_test_df` helper in `routes.py` mirrors `clean_dataset` from `server.py` with three deliberate omissions: no duplicate dropping (test rows are independent predictions), no row dropping for missing target values (there is no target column in test data), and no constant column dropping (a column constant in the test set may not have been constant in training). Applying the full `clean_dataset` logic to test data would silently alter the row count and break the prediction output.


---
## Frontend Design and UI

The frontend is a Gradio app with two tabs: Train (upload a CSV, specify a target column, run the full AutoML workflow) and Predict (select a model type, upload a test CSV or trigger a forecast, download results).

### Why Gradio?
Gradio was chosen for speed of iteration rather than UI flexibility. A React frontend would have given more control over layout and state management, but the primary goal was end-to-end system integration, not frontend engineering. Gradio's `gr.State` components handle the cross-tab state problem — `job_id`, `train_filename`, and `problem_type` are persisted in hidden state components and passed automatically to the `Predict` tab's handlers.

### Dynamic UI on problem type
After a training run completes, the frontend receives `problem_type` in the `RunResponse` and immediately reconfigures the `Predict` tab: if the problem type is forecasting, the test file upload widget is hidden and an image output component is shown instead (since the forecasting predict path returns a PNG plot rather than a CSV). The model type dropdown is also repopulated with `["lstm", "transformer"]` for forecasting and the full `sklearn+FFN` list otherwise. This is done via `gr.update()` calls returned from the `upload_and_run` function — Gradio's mechanism for imperatively modifying component state from a Python callback.

### Forecasting predict path
For forecasting, the frontend sends a predict request with an empty `test_filename` — the backend doesn't need a test file because the test split was saved at training time inside the `.npz` file. The backend returns a `FileResponse` with `media_type="image/png"`, and the frontend writes the raw bytes to `/tmp/forecast_plot.png` and passes the path to `gr.Image`. This is a deliberate asymmetry with the tabular predict path (which returns a CSV), handled by returning `(status, None, image_path)` vs `(status, csv_path, None)` from the predict function.

---
### Deployment

`TensorFlow` imports increase the total size of the Dockerized images to 8GB; as such, the app was not deployed on HF spaces. 

----
## References
* **Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers**, Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Gopi Krishnan Rajbahadur, Bram Adams, Ahmed E. Hassan. (2026). [arXiv:2506.13538v5](https://arxiv.org/abs/2506.13538).
* **Model Context Protocol Explained in 3 Levels of Difficulty**, Bala Priya C. (2026). [(here)](https://machinelearningmastery.com/model-context-protocol-explained-in-3-levels-of-difficulty/)
* **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML**, Patara Trirat, Wonyong Jeong, Sung Ju Hwang. (2024). [arXiv:2410.02958](https://arxiv.org/abs/2410.02958).





