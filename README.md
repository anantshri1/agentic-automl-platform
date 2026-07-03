# Agentic AutoML Platform

The Agentic AutoML Platform is an end-to-end machine learning system that automates the full lifecycle of tabular ML workflows using an LLM-driven orchestration layer. Instead of treating AutoML as a black-box model selection tool, this system decomposes the ML pipeline into modular, composable tools exposed via an MCP (Model Context Protocol) server and orchestrated through a LangGraph-based agent.
 
The system allows users to upload structured datasets (CSV) and interact in natural language to perform data analysis, model training, evaluation, and experimentation. The core idea is to shift AutoML from static optimization into an agentic workflow system with observability, control, and extensibility.

Key Features
1. Natural Language ML Workflow Interface
 
Users can describe tasks such as:
 
“Predict house prices”
“Which model works best for this dataset?”
“What features matter most?”
“Compare XGBoost vs Random Forest”
 
The agent translates these into structured ML workflows.
 
2. Agentic AutoML Engine (LangGraph-based)
 
The system uses a graph-based reasoning engine that dynamically constructs ML pipelines:
 
Typical workflow:
 
Dataset ingestion
Problem type detection (classification / regression / forecasting)
Data profiling
Missing value & feature analysis
Feature engineering suggestions
Model selection
Training
Hyperparameter tuning
Evaluation
Explanation & reporting
 
Unlike traditional AutoML systems, each stage is explicitly visible and controllable.
 
3. MCP Tooling Layer
 
All ML capabilities are exposed as tools via an MCP server. This enables modular execution and extensibility.
 
Core tools include:
 
profile_dataset()
detect_problem_type()
clean_dataset()
suggest_features()
train_model(model_type, params)
hyperparameter_search(config)
evaluate_model(metrics)
feature_importance()
generate_report()
 
The LLM does not execute ML logic directly; it orchestrates tools via MCP calls.
 
4. FastAPI Backend
 
A production-style backend exposes the system via REST APIs:
 
Endpoints include:
 
POST /upload → upload dataset
POST /run → start ML workflow
GET /job/{id} → job status & logs
GET /results/{id} → predictions + metrics
GET /explain/{id} → model explanations
 
The backend handles:
 
job scheduling
state management
caching intermediate results
tool routing via MCP
5. Experiment Tracking (MLflow Integration)
 
Every run is fully tracked:
 
model parameters
metrics (accuracy, RMSE, F1, etc.)
feature sets
artifacts (models, plots)
training logs
 
This enables reproducibility and comparison across experiments.
 
6. Observability Layer (LangSmith)
 
The system is instrumented with LangSmith to track:
 
agent decision paths
tool calls
latency per step
prompt versions
failure modes
 
This provides full transparency into the agent’s reasoning process.
 
7. Dockerized Architecture
 
The entire system is containerized for reproducibility and deployment:
 
Services:
 
frontend (Gradio or lightweight UI)
backend (FastAPI)
MCP server
MLflow tracking server
optional database (PostgreSQL / SQLite)
 
Managed via docker-compose for one-command setup.
 
8. Human-in-the-loop Control (Optional)
 
The system can request user feedback during pipeline execution:
 
Examples:
 
confirm feature removal
approve model selection
choose evaluation metric
override preprocessing decisions
 
This ensures interpretability and control over automated decisions.
 
System Architecture
 
User Interface
↓
FastAPI Backend
↓
LangGraph Orchestrator
↓
MCP Client
↓
MCP Tool Server
↓
ML Modules:
 
Data Profiler
Feature Engineering
Model Trainer
Hyperparameter Optimizer
Evaluator
Explainer
↓
MLflow Tracking
↓
Artifacts + Metrics Storage
Supported Problem Types
 
The system automatically detects and supports:
 
Regression
Classification
Time-series forecasting (experimental extension, no nixtla, maybe XGBoost, lstm)
 
Design Philosophy
 
This system is built on three principles:
 
1. Decomposition over abstraction
 
Instead of hiding ML behind a single AutoML call, each step is explicitly modelled and executable.
 
2. Agentic orchestration over static pipelines
 
The system adapts workflows dynamically based on dataset properties and user intent.
 
3. Observability over black-box automation
 
Every decision made by the system is logged, traceable, and inspectable.
 
Example Workflow
 
User:
“I want to predict customer churn.”
 
System:
 
Detects classification problem
Profiles dataset
Finds missing values in tenure column
Suggests imputation strategy
Trains baseline models:
Logistic Regression
XGBoost
Random Forest
Runs hyperparameter tuning on best model
Evaluates using F1-score
Generates feature importance report
Returns best model + explanation
Technology Stack
Python
FastAPI
LangGraph
MCP (Model Context Protocol)
MLflow
Docker / Docker Compose
scikit-learn / XGBoost / LightGBM
Pandas / NumPy
LangSmith (observability)
Optional: React / Gradio frontend
Deployment
 
The system is designed for deployment as:
 
Local Docker Compose stack
Hugging Face Spaces (lightweight frontend variant)
Cloud deployment (AWS / GCP) with scalable backend workers
