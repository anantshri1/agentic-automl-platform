# Agentic AutoML Platform
> **(Or, the time I used gradient descent to destroy the gradient descent)**

The Agentic AutoML Platform is an end-to-end machine learning system that automates the full lifecycle of tabular ML workflows using an LLM-driven orchestration layer. Instead of treating AutoML as a black-box model selection tool, this system decomposes the ML pipeline into modular, composable tools exposed via an MCP (Model Context Protocol) server and orchestrated through a LangGraph-based agent.
 
The system allows users to upload structured datasets (CSV) and interact in natural language to perform data analysis, model training, evaluation, and experimentation. The core idea is to shift AutoML from static optimization into an agentic workflow system with observability, control, and extensibility.

**Key Features**
1. **Natural Language ML Workflow Interface**: Users can describe tasks such as: “Predict house prices”, “Compare XGBoost vs Random Forest”; the agent translates these into structured ML workflows.
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
 
5. **Experiment Tracking (MLflow Integration)**: Every run is fully tracked:
 - model parameters
 - metrics (accuracy, RMSE, F1, etc.)
 - feature sets
 - artifacts (models, plots)
 - training logs

This enables reproducibility and comparison across experiments.

6. **Observability Layer (LangSmith)**: The system is instrumented with LangSmith to track agent decision paths, tool calls, latency per step, prompt versions, and failure modes. This provides full transparency into the agent’s reasoning process. 
7. **Dockerized Architecture**: The entire system is containerized for reproducibility and deployment.
 
---
## Backend Design (via `FastAPI`)



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

----
## Backend CURL commands

uploading new file:
```
curl -X POST http://localhost:8000/upload -F "file=@irrigation_prediction.csv"
```

running algorithm:
```
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "4a09744a-5a9b-4154-93dc-740e8ef51a8d",
    "filename": "irrigation_prediction.csv",
    "target_column": "Irrigation_Need"
  }'
```

finding files:
```
docker exec -it automl-platform-backend-1 ls /app/data/ | grep 4a09744a-5a9b-4154-93dc-740e8ef51a8d
```

prediction:
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



