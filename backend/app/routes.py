""" The plan for `POST /upload`:
1. Receive a CSV file from the client.
2. Generate a unique job ID (UUID).
3. Save the CSV (inside the container, via volume mount)
4. Return an `UploadResponse` with the job ID, filename, and status.

> For UUID: Python has build-in `uuid` module. `str(uuid.uuid4())` generates a random UUID string.
> For saving files: FastAPI provides `UploadFile` which is a wrapper around `starlette.datastructures.UploadFile`. It has a `.file` attribute which is a file-like object. You can read from it and write to disk.
"""

""" What's happening here?
* `APIRouter` — a mini FastAPI app. We define routes here and register them in `main.py`. Keeps things modular
* `UploadFile = File(...)` — FastAPI's way of saying "expect a file in this request". The `...` means it's required
* `uuid.uuid4()`— generates a random unique ID for every upload
* `shutil.copyfileobj` — streams the file to disk efficiently without loading it all into memory
* `UPLOAD_DIR = Path("/app/data")` — this is the path inside the container, which maps to `backend/data/` on your Mac via the volume mount
* `RunRequest` comes in as a JSON body, FastAPI automatically parses it into a Python object thanks to Pydantic.
* `queued` is a placeholder status. In a real system, you'd enqueue the job for processing and return a job ID immediately.
"""

""" Moving Parts:
* `POST /upload` — receives a CSV, saves it, returns job ID
* `POST /run` — receives a job ID and target column, returns status
* `GET /job/{job_id}` — returns the status of a job (queued, running, completed, failed)
* `GET /results/{job_id}` — returns the results of a completed job (metrics, predictions)
"""

from email.mime import base
from urllib import request
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models import UploadResponse
from app.models import RunRequest, RunResponse, JobStatus, ResultsResponse
from app.models import PredictRequest
import os
import json
import httpx
import pandas as pd
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from fastapi.responses import FileResponse

# ── Transformer custom layers ────────────────────────────────────────────────
import numpy as np
import tensorflow as tf

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense   = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key   = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.split_heads(query, batch_size)
        key   = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        attention_output, _ = self.attention(query, key, value)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads})
        return config


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.embed_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, embed_dim,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        p, i = np.meshgrid(np.arange(max_len), np.arange(embed_dim // 2))
        angles = p / np.power(10000, 2 * i / embed_dim)
        pos_encoding = np.zeros((max_len, embed_dim))
        pos_encoding[:, 0::2] = np.sin(angles.T)
        pos_encoding[:, 1::2] = np.cos(angles.T)
        self.pos_encoding = tf.cast(pos_encoding[np.newaxis, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.pos_encoding.shape[1], "embed_dim": self.embed_dim})
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1,**kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.enc_layers = [
            TransformerBlock(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        x = self.dropout(inputs, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": len(self.enc_layers),
            "embed_dim": self.enc_layers[0].att.embed_dim,
            "num_heads": self.enc_layers[0].att.num_heads,
            "ff_dim": self.enc_layers[0].ffn.layers[0].units,
            "rate": self.dropout.rate,
        })
        return config


""" MCP Client API Calls:
Calling an MCP tool over HTTP isn't a plain `requests.post()`. The `mcp` package has its own async client that does a proper protocol handshake — it connects, asks the server "what tools do you have?" (`list_tools`), then calls the one you want (`call_tool`). This happens every time you open a `ClientSession`. It's stateless from our perspective — no persistent connection to manage.

`streamablehttp_client` is an async context manager that opens the HTTP connection. 
`ClientSession` wraps it and gives you `call_tool()`.
"""

"""
We're opening a fresh MCP client session per request. Not the most efficient thing in the world (a connection pool would be better), but correct and simple for now. 
We'll flag this as a known limitation.
"""

async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    mcp_url = os.getenv("MCP_SERVER_URL", "http://mcp-server:8001") + "/mcp"     #  reads from .env with a sensible fallback.
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return json.loads(result.content[0].text)
            # MCP returns a list of content blocks. Our stub tools return dicts, which FastMCP serializes as JSON text in the first block

router = APIRouter()

UPLOAD_DIR = Path("/app/data")


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    destination = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return UploadResponse(
        job_id=job_id,
        filename=f"{job_id}_{file.filename}",
        status="uploaded"
    )

"""
What's new here:
* `message` — we translate the structured `RunRequest` into natural language. This is what the LLM actually reasons over. The job_id and target column are embedded in the text so the agent can pass them to tools.
* `httpx.AsyncClient(timeout=60.0)` — async HTTP client, used as a context manager. The 60 second timeout is intentional: the ReAct loop makes multiple LLM calls and tool calls, so it's slower than a direct MCP call. 
* `resp.raise_for_status()` — if the orchestrator returns a 4xx or 5xx, this raises an exception immediately rather than silently returning bad data.
"""
@router.post("/run", response_model=RunResponse)
async def run_workflow(request: RunRequest):
    # Detect problem type directly — don't rely on extracting it from agent text
    detection = await call_mcp_tool("detect_problem_type", {
        "dataset_id": f"{request.job_id}_{request.filename}",
        "target_column": request.target_column,
    })
    detected_problem_type = detection.get("problem_type", "")

    message = (
        f"I have a dataset with filename '{request.job_id}_{request.filename}'. "
        f"The target column is '{request.target_column}'. "
        f"Follow these steps:\n"
        f"1. Profile the dataset using profile_dataset.\n"
        f"2. Detect the problem type using detect_problem_type.\n"
        f"3. If problem_type is 'forecasting':\n"
        f"   a. Clean the dataset using clean_dataset with problem_type='forecasting'.\n"
        f"   b. Call prepare_forecast_dataset with an appropriate window_size (default 30) "
        f"and the datetime column detected during profiling.\n"
        f"   c. Train both train_lstm and train_transformer using the npz_id returned.\n"
        f"   d. Report both model_ids, run_ids, and final_metrics for comparison.\n"
        f"4. If problem_type is 'classification' or 'regression':\n"
        f"   a. Clean the dataset using clean_dataset.\n"
        f"   b. Check class balance using check_class_balance if classification.\n"
        f"   c. Run reduce_dimensions if clean_dataset flagged high dimensionality.\n"
        f"   d. Run hyperparameter_search FOUR times — once for each model type: "
        f"logistic_regression, random_forest, xgboost, catboost. "
        f"Pass the same dataset_id, target_column, and problem_type each time.\n"
        f"   e. Train ALL supported model types using train_model: "
        f"logistic_regression (classification) or linear_regression (regression), "
        f"random_forest, xgboost, and catboost. Use best_params from hyperparameter_search "
        f"where the model type matches, default params otherwise.\n"
        f"   f. Evaluate each trained model using evaluate_model.\n"
        f"   g. Compare test metrics across all models and clearly state which performed best and why.\n"
        f"   h. Always call train_ffn after the sklearn models, regardless of their performance.\n"
        f"   i. Immediately after train_ffn, call evaluate_ffn with the same dataset_id, "
        f"target_column, problem_type, and the run_id returned by train_ffn. "
        f"Report its test metrics and compare with the best sklearn model.\n"
        f"j. After every tool call, check whether the result contains 'error'."
        f"If it does, stop and report the failure instead of continuing with invalid paths."
    )

    async with httpx.AsyncClient(timeout=200.0) as client:
        resp = await client.post(
            "http://orchestrator:8002/invoke",
            json={"message": message}
        )
        resp.raise_for_status()
        agent_response = resp.json()["response"]

    return RunResponse(
        job_id=request.job_id,
        status="completed",
        filename=f"{request.job_id}_{request.filename}".replace(".csv", "_cleaned.csv"),
        results={"agent_response": agent_response},
        problem_type=detected_problem_type
    )


@router.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    return JobStatus(
        job_id=job_id,
        status="queued",
        logs=["Job received", "Waiting for worker"]
    )

@router.get("/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str):
    return ResultsResponse(
        job_id=job_id,
        metrics=None,
        predictions=None
    )

@router.post("/predict")
async def predict(request: PredictRequest):
    test_path = f"/app/data/{request.test_filename}"
    model_type = request.model_type
    problem_type = request.problem_type

    # base_cleaned: stem for all model artifacts (sklearn, ffn, pca)
    # covers both "123_abc_cleaned" and "123_abc_cleaned_pca" cases
    base_cleaned = request.train_filename.replace(".csv", "")

    # base_raw: stem for encoders and label map
    # these were saved by clean_dataset using the original raw filename
    base_raw = base_cleaned.split("_cleaned")[0]
    
    
    def _clean_test_df(df):
        import re
        import pandas as pd
        # Step 1b: Drop ID columns by name pattern only (no uniqueness check — test size differs)
        id_cols = [
            col for col in df.columns
            if col == "id" or col.endswith("_id") or col.startswith("id_")
        ]
        if id_cols:
            df.drop(columns=id_cols, inplace=True)

        # Step 1c: Combine year/month/day into single date column
        has_year  = "year"  in df.columns
        has_month = "month" in df.columns
        has_day   = "day"   in df.columns
        if has_year and has_month:
            df["date"] = pd.to_datetime({
                "year":  df["year"],
                "month": df["month"],
                "day":   df["day"] if has_day else 1,
            }, errors="coerce")
            cols_to_drop = ["year", "month"] + (["day"] if has_day else [])
            df.drop(columns=cols_to_drop, inplace=True)

        # Step 2: Fix datetime columns
        for col in df.columns:
            if any(kw in col for kw in ["date", "time", "timestamp"]):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / len(df) >= 0.8:
                    df[col] = parsed

        # Step 2b: Coerce mistyped numeric columns
        datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        for col in df.columns:
            if col in datetime_cols:
                continue
            if df[col].dtype == "object":
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().sum() / len(df) >= 0.8:
                    df[col] = coerced

        # Step 5: Impute nulls — no row dropping (test rows are predictions, not training samples)
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown", inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        return df
    
    if problem_type == "forecasting":
        import pickle
        import numpy as np
        import tensorflow as tf
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — no display needed inside Docker
        import matplotlib.pyplot as plt
        import json

        # --- Block 1: Load meta ---
        # meta tells us window_size, target_column, feature_columns, n_features
        # all saved by prepare_forecast_dataset at training time
        meta_path = f"/app/data/{base_raw}_forecast_meta.json"
        if not os.path.exists(meta_path):
            raise HTTPException(status_code=404, detail=f"Forecast meta not found at {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        target_column   = meta["target_column"]
        feature_columns = meta["feature_columns"]
        n_features      = meta["n_features"]
        window_size     = meta["window_size"]

        # --- Block 2: Load windowed test arrays ---
        # X_test shape: (n_test, window_size, n_features) — already scaled
        # y_test shape: (n_test,)                         — already scaled
        npz_path = f"/app/data/{base_raw}_windows.npz"
        if not os.path.exists(npz_path):
            raise HTTPException(status_code=404, detail=f"Windows .npz not found at {npz_path}")

        data   = np.load(npz_path)
        X_test = data["X_test"]
        y_test = data["y_test"]

        # --- Block 3: Load model ---
        # model_type is either "lstm" or "transformer"
        model_path = f"/app/data/{base_raw}_{model_type}.keras"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

        # Transformer has custom layers — we must pass them to load_model
        # LSTM has no custom objects so the same call works for both
        custom_objects = {
            "MultiHeadSelfAttention": MultiHeadSelfAttention,
            "TransformerBlock": TransformerBlock,
            "PositionalEncoding": PositionalEncoding,
            "TransformerEncoder": TransformerEncoder,
        }
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model at {model_path}: {type(e).__name__}: {e}")

        # --- Block 4: Load target scaler and inverse-transform ---
        # Predictions and actuals are both in scaled [0,1] space
        # We inverse-transform both so the plot is in real units
        scaler_path = f"/app/data/{base_raw}_forecast_scaler.pkl"
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Target scaler not found at {scaler_path}")

        with open(scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

        y_pred_scaled = model.predict(X_test).flatten()

        # inverse_transform expects shape (n, 1)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # --- Block 5: Plot actual vs predicted ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_true, label="Actual",    color="steelblue",  linewidth=1.5)
        ax.plot(y_pred, label="Predicted", color="darkorange", linewidth=1.5, linestyle="--")
        ax.set_title(f"Actual vs Predicted — {target_column} ({model_type.upper()})")
        ax.set_xlabel("Time step")
        ax.set_ylabel(target_column)
        ax.legend()
        fig.tight_layout()

        plot_path = f"/app/data/{request.job_id}_{model_type}_forecast.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)   # free memory — important in a long-running server

        return FileResponse(
            path=plot_path,
            media_type="image/png",
            filename=f"{request.job_id}_{model_type}_forecast.png"
        )

    # Branch on model_type — logic coming in Block 2 and 3
    sklearn_models = {"logistic_regression","linear_regression", "random_forest", "xgboost", "catboost"}
    if model_type in sklearn_models:
        import pickle
        import pandas as pd

        if not os.path.exists(test_path):
            raise HTTPException(status_code=404, detail=f"Test file not found at {test_path}")

        model_path = f"/app/data/{base_cleaned}_{model_type}.pkl"
        pca_applied = False
        if not os.path.exists(model_path):
            pca_model_path = f"/app/data/{base_cleaned}_pca_{model_type}.pkl"
            if os.path.exists(pca_model_path):
                model_path = pca_model_path
                pca_applied = True
            else:
                raise HTTPException(status_code=404, detail=f"Model not found at {model_path} or {pca_model_path}")

        with open(model_path, "rb") as f:
            pipe = pickle.load(f)

        df = pd.read_csv(test_path)

        # Step 1: Standardise column names — mirrors clean_dataset Step 1
        import re
        new_columns = {}
        for col in df.columns:
            new_col = col.lower().strip()
            new_col = re.sub(r'[\s\W]+', '_', new_col)
            new_col = re.sub(r'_+', '_', new_col)
            new_col = new_col.strip('_')
            new_columns[col] = new_col
        df.rename(columns=new_columns, inplace=True)
        df = _clean_test_df(df)

        # Step 2: Apply fitted label encoders — mirrors clean_dataset Step 7
        encoders_path = f"/app/data/{base_raw}_encoders.pkl"
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                feature_encoders = pickle.load(f)
            print("encoder keys:", list(feature_encoders.keys()))
            print("test df columns:", list(df.columns))
            for col, le in feature_encoders.items():
                if col in df.columns:
                    known_classes = set(le.classes_)

                    df[col] = (
                        df[col]
                        .astype(str)
                        .apply(lambda x: x if x in known_classes else le.classes_[0])
                    )

                    df[col] = le.transform(df[col])
        else:
            print("NO ENCODERS FILE FOUND at", encoders_path)

        # Apply PCA transform if it was used during training
        pca_pkl_path = f"/app/data/{base_cleaned}_pca.pkl"
        if pca_applied and os.path.exists(pca_pkl_path):
            import pickle as pkl
            with open(pca_pkl_path, "rb") as f:
                pca = pkl.load(f)
            df = pd.DataFrame(
                pca.transform(df),
                columns=[f"pc_{i}" for i in range(pca.n_components_)]
            )
        predictions = pipe.predict(df).flatten().tolist()
        label_map_path = f"/app/data/{base_raw}_label_map.json"
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
                #print("label_map:", label_map)
                #print("sample predictions:", predictions[:3])
            predictions = [label_map.get(str(p), p) for p in predictions]
            

        output_path = f"/app/data/{request.job_id}_predictions.csv"
        pd.DataFrame({"predictions": predictions}).to_csv(output_path, index=False)
    elif model_type == "ffn":
        import pickle
        import numpy as np
        import tensorflow as tf
        import pandas as pd

        if not os.path.exists(test_path):
            raise HTTPException(status_code=404, detail=f"Test file not found at {test_path}")

        scaler_path = f"/app/data/{base_cleaned}_ffn_scaler.pkl"
        model_path  = f"/app/data/{base_cleaned}_ffn.keras"
        pca_applied = False
        if not os.path.exists(model_path):
            pca_model_path = f"/app/data/{base_cleaned}_pca_ffn.keras"
            pca_scaler_path = f"/app/data/{base_cleaned}_pca_ffn_scaler.pkl"
            if os.path.exists(pca_model_path):
                model_path = pca_model_path
                scaler_path = pca_scaler_path
                pca_applied = True
            else:
                raise HTTPException(status_code=404, detail=f"Model not found at {model_path} or {pca_model_path}")

        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Scaler not found at {scaler_path}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Scaler not found at {scaler_path}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise HTTPException(status_code=404, detail=f"Scaler not found at {scaler_path}")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model not found at {model_path}")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model = tf.keras.models.load_model(model_path)

        df = pd.read_csv(test_path)
        # Standardise column names — mirrors clean_dataset Step 1
        import re
        new_columns = {}
        for col in df.columns:
            new_col = col.lower().strip()
            new_col = re.sub(r'[\s\W]+', '_', new_col)
            new_col = re.sub(r'_+', '_', new_col)
            new_col = new_col.strip('_')
            new_columns[col] = new_col
        df.rename(columns=new_columns, inplace=True)
        df = _clean_test_df(df)

        # Apply fitted label encoders — mirrors clean_dataset Step 7
        encoders_path = f"/app/data/{base_raw}_encoders.pkl"
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                feature_encoders = pickle.load(f)
            for col, le in feature_encoders.items():
                if col in df.columns:
                    df[col] = le.transform(df[col].astype(str))
        X = df.values.astype(float)

        pca_pkl_path = f"/app/data/{base_cleaned}_pca.pkl"
        if pca_applied and os.path.exists(pca_pkl_path):
            import pickle as pkl
            with open(pca_pkl_path, "rb") as f:
                pca = pkl.load(f)
            X = pca.transform(X)

        X = scaler.transform(X)

        raw_preds = model.predict(X)
        output_shape = raw_preds.shape

        if problem_type == "classification":
            if output_shape[1] == 1:
                predictions = (raw_preds > 0.5).astype(int).flatten().tolist()
            else:
                predictions = np.argmax(raw_preds, axis=1).tolist()
        else:
            predictions = raw_preds.flatten().tolist()

        label_map_path = f"/app/data/{base_raw}_label_map.json"
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path, "r") as f:
                label_map = json.load(f)
            predictions = [label_map.get(str(p), p) for p in predictions]

        output_path = f"/app/data/{request.job_id}_predictions.csv"
        pd.DataFrame({"predictions": predictions}).to_csv(output_path, index=False)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model_type: {model_type}")
    return FileResponse(
        path=output_path,
        media_type="text/csv",
        filename=f"{request.job_id}_predictions.csv"
    )

