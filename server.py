""" How `StreamableHTTP` MCP works:
The `mcp` Python package lets you define a server like this:
```
mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(x: int) -> str:
    ...

mcp.run(transport="streamable-http")
```
Internally, FastMCP with streamable-http spins up a Starlette app (a lightweight ASGI framework, same family as FastAPI) and serves it over HTTP. Your tools become callable endpoints that speak the MCP protocol — not plain `REST`, but a structured JSON protocol over HTTP.
The client side (FastAPI, later) uses mcp's async client to connect to that URL and call tools by name. It's not a `requests.get()` — it's the MCP client handshaking, listing tools, and invoking them by name.

"""

"""
`FastMCP` is the high-level class from the `mcp` package. You instantiate it with a name, decorate Python functions with `@mcp.tool()`, and it handles all the protocol machinery — tool listing, schema generation from type hints, JSON serialization. You just write normal Python functions.
The `dict` return types matter: MCP tools return structured data, and the type hints are what `FastMCP` uses to auto-generate the tool schema that clients can inspect.
"""
from logging import config
import os
import re
import tensorflow as tf
import numpy as np
import pickle

# mcp_server/server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("automl-tools",host = "0.0.0.0" ,port = 8001)

""" Design notes on `profile_dataset`:

**What this tool needs to do**
Given a `dataset_id` (which is just the filename the backend saved to `/app/data/`), the tool should:
* Construct the full path: `/app/data/{dataset_id}`
* Load it with pandas
* Return: row/column counts, dtypes per column, missing value counts per column, and basic descriptive stats (mean, std, min, max) for numeric columns

> Why `orient="index"` on descriptive stats
`df.describe()` returns a DataFrame, not a plain dict. If you call `.to_dict()` on it directly you get `{"mean": {"age": 32.1, ...}, "std": {...}}` — outer keys are stat names, inner keys are columns. That's awkward to read. `df.describe().to_dict(orient="index")` flips it: outer keys are stat names (count, mean, std...), still nested but more natural when you're scanning per-stat. We'll use that.
"""
@mcp.tool()
def profile_dataset(dataset_id: str) -> dict:
    """Read an uploaded CSV and return a real statistical profile."""
    import pandas as pd

    path = f"/app/data/{dataset_id}"

    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)

    # missing value count per column, only include columns that have at least one
    missing = {
        col: int(count)
        for col, count in df.isnull().sum().items()
        if count > 0
    }

    # dtypes as strings so they're JSON-serialisable
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # descriptive stats for numeric columns only
    # .to_dict(orient="index") → {"count": {"age": 1000.0, ...}, "mean": {...}, ...}
    stats = df.describe().to_dict(orient="index")

    # flag columns with values beyond 3 standard deviations
    outlier_flags = {}
    for col in df.select_dtypes(include="number").columns:
        mean, std = df[col].mean(), df[col].std()
        if std == 0:
            continue
        n_outliers = int(((df[col] - mean).abs() > 3 * std).sum())
        if n_outliers > 0:
            outlier_flags[col] = {
                "count": n_outliers,
                "pct": round(n_outliers / len(df) * 100, 2)
            }

    return {
        "dataset_id": dataset_id,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing,
        "descriptive_stats": stats,
        "outlier_flags": outlier_flags,
    }
    
""" Design notes on `detect_problem_type`:
The heuristic has three steps, applied to the target column:
Step 1 — dtype check. If the column's dtype is `object` or `category`, it's categorical by definition → classification.
Step 2 — unique value count (cardinality check). If dtype is numeric but there are few unique values (≤ 10), it's almost certainly a label encoded classification target (e.g. 0/1 for churn, 1/2/3 for star ratings).
Step 3 — otherwise regression. Continuous numeric column with high cardinality → regression.
This won't be right 100% of the time — a column of years (1990–2024) is numeric with ~34 unique values and would be called regression, which may or may not be what you want. But it's the right heuristic for a first pass, and the agent can always let the user override it.
> The tool returns a reasoning string explaining why it made the call. This isn't just cosmetic — when LangGraph reads tool outputs to decide next steps, having an explicit reasoning field means the agent can log or surface it to the user. It makes the system inspectable.
"""
@mcp.tool()
def detect_problem_type(dataset_id: str, target_column: str) -> dict:
    """Inspect the target column and return classification or regression."""
    import pandas as pd

    path = f"/app/data/{dataset_id}"

    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)

    # Forecasting detection: check non-target columns for datetime name heuristics
    datetime_keywords = {"date", "time", "timestamp", "year", "month"}
    feature_cols = [c for c in df.columns if c != target_column]
    datetime_cols = []
    for c in feature_cols:
        if not any(kw in c.lower() for kw in datetime_keywords):
            continue
        # Must parse as actual datetime values
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            parseable_ratio = parsed.notna().sum() / len(df)
            if parseable_ratio < 0.8:
                continue
        except Exception:
            continue
        # Must be high cardinality — acting as a time index not a categorical
        uniqueness_ratio = df[c].nunique() / len(df)
        if uniqueness_ratio < 0.5:
            continue
        datetime_cols.append(c)
    
    if datetime_cols:
        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": "forecasting",
            "datetime_columns_found": datetime_cols,
            "reasoning": f"Found datetime-like columns {datetime_cols} → forecasting.",
            "note": "Use clean_dataset (forecasting mode) → prepare_forecast_dataset → train_lstm or train_transformer.",
        }
    
    if target_column not in df.columns:
        return {"error": f"Column '{target_column}' not found. Available: {list(df.columns)}"}

    col = df[target_column]
    n_unique = int(col.nunique())
    dtype_str = str(col.dtype)

    # Step 1: categorical dtype → classification
    if col.dtype == "object" or str(col.dtype) == "category":
        problem_type = "classification"
        reasoning = f"Target dtype is '{dtype_str}' (categorical) → classification."

    # Step 2: numeric but low cardinality → classification
    elif n_unique <= 10:
        problem_type = "classification"
        reasoning = f"Target is numeric with only {n_unique} unique values → treating as classification."

    # Step 3: continuous numeric → regression
    else:
        problem_type = "regression"
        reasoning = f"Target is numeric with {n_unique} unique values → treating as regression."

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "dtype": dtype_str,
        "n_unique": n_unique,
        "problem_type": problem_type,
        "reasoning": reasoning,
    }

""" Design Notes on EDA strategies:
* Column name standardisation
```
lowercase → strip whitespace → replace spaces/special chars with underscore
``
* Dtype fixing
```
any column with "date" or "time" in the name → pd.to_datetime(), errors="coerce"
```

```
for each object-dtype column that isn't target and isn't datetime:
    try pd.to_numeric(col, errors="coerce")
    if the result has ≥ 80% non-null values → accept the conversion, flag it
    otherwise → leave as-is (it's probably genuinely categorical)
```
* Null handling (per column, excluding target)
```
missing < 5%   → drop rows where that column is null
missing 5–30%  → numeric: median impute / categorical: mode impute
missing > 30%  → flag in output, leave column as-is (agent decides)
```
* Duplicates
```
df.drop_duplicates(), report how many were dropped
```
* Target encoding
```
if problem_type == "classification" and target dtype is object/category:
    LabelEncoder → save mapping to /app/data/{dataset_id}_label_map.json
```
* Feature encoding
```
categorical column with ≤ 10 unique values → one-hot encode
categorical column with > 10 unique values → flag, leave as-is (agent decides)
```
* Output
```
save cleaned df to /app/data/{dataset_id}_cleaned.csv
save label map to /app/data/{dataset_id}_label_map.json (if classification)
return: summary of all actions taken + any flags
```

"""
@mcp.tool()
def clean_dataset(dataset_id: str, target_column: str, problem_type: str) -> dict:
    """Clean and preprocess a dataset. Returns path to cleaned CSV and action summary."""
    import pandas as pd
    import json

    path = f"/app/data/{dataset_id}"        # path = /app/data/123_abc.csv
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)
    actions = []
    flags = []

    # --- Step 1: Standardise column names ---
    # Build a mapping of old → new names so we can report what changed
    new_columns = {}
    for col in df.columns:
        new_col = col.lower().strip()
        new_col = re.sub(r'[\s\W]+', '_', new_col)  # spaces and non-word chars → underscore
        new_col = re.sub(r'_+', '_', new_col)        # collapse multiple underscores
        new_col = new_col.strip('_')                  # strip leading/trailing underscores
        new_columns[col] = new_col

    changed = {old: new for old, new in new_columns.items() if old != new}
    if changed:
        df.rename(columns=new_columns, inplace=True)
        actions.append(f"Standardised column names: {changed}")

    # target_column may have been renamed — update it to match
    target_column = new_columns.get(target_column, target_column)

    # --- Step 1b: Drop ID columns ---
    # Heuristic: name matches id pattern AND every row is unique (true ID, not a categorical)
    id_cols = [
        col for col in df.columns
        if col != target_column
        and (col == "id" or col.endswith("_id") or col.startswith("id_"))
        and df[col].nunique() == len(df)
    ]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        actions.append(f"Dropped ID columns (unique per row): {id_cols}")
    
    # --- Step 1c: Combine separate year/month/day columns into single datetime ---
    # Handles datasets where datetime is split across multiple columns
    # Day defaults to 1 if missing (safe for monthly time series)
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
        actions.append(
            f"Combined {cols_to_drop} into single 'date' column."
        )

    # --- Step 2: Fix datetime columns ---
    for col in df.columns:
        if col == target_column:
            continue
        if any(kw in col for kw in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            parseable_ratio = parsed.notna().sum() / len(df)
            if parseable_ratio >= 0.8:
                df[col] = parsed
                actions.append(f"Converted '{col}' to datetime.")
    
     # --- Step 2b: Coerce mistyped numeric columns ---
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    for col in df.columns:
        if col == target_column or col in datetime_cols:
            continue
        if df[col].dtype == "object":
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_null_ratio = coerced.notna().sum() / len(df)
            if non_null_ratio >= 0.8:
                df[col] = coerced
                actions.append(f"Coerced '{col}' from object to numeric ({non_null_ratio:.0%} valid values).")


    # --- Step 3: Drop constant columns ---
    constant_cols = [
        col for col in df.columns
        if col != target_column and df[col].nunique() == 1
    ]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        actions.append(f"Dropped constant columns (zero variance): {constant_cols}")

    # --- Step 4: Drop duplicates ---
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        actions.append(f"Dropped {dropped} duplicate rows.")

    # --- Step 5: Null handling (per column, excluding target) ---
    for col in df.columns:
        if col == target_column:
            continue
        missing_pct = df[col].isnull().sum() / len(df)
        if missing_pct == 0:
            continue
        elif missing_pct < 0.05:
            before = len(df)
            df.dropna(subset=[col], inplace=True)
            actions.append(f"'{col}': {missing_pct:.1%} missing → dropped {before - len(df)} rows.")
        elif missing_pct <= 0.30:
            if df[col].dtype == "object":
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                actions.append(f"'{col}': {missing_pct:.1%} missing → mode imputed ('{fill_val}').")
            else:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                actions.append(f"'{col}': {missing_pct:.1%} missing → median imputed ({fill_val:.4g}).")
        else:
            flags.append(f"'{col}': {missing_pct:.1%} missing — left as-is, review recommended.")
    
    # --- Step 5b: Drop rows with missing target ---
    before = len(df)
    df.dropna(subset=[target_column], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        actions.append(f"Dropped {dropped} rows with missing target '{target_column}'.")

    # --- Step 6: Target encoding (classification only) ---
    label_map = {}
    if problem_type == "classification" and df[target_column].dtype == "object":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])
        label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
        actions.append(f"Label-encoded target '{target_column}': {label_map}")

    # --- Step 7: Feature encoding ---
    # Encode categorical features for all problem types.
    # Forecasting may also contain categorical covariates (product, region, etc.)

    from sklearn.preprocessing import LabelEncoder
    encoders_path = None
    cols_to_label_encode = [
        col for col in df.columns
        if col != target_column and df[col].dtype == "object"
    ]

    feature_encoders = {}

    for col in cols_to_label_encode:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        feature_encoders[col] = le
        actions.append(f"Label-encoded feature column '{col}'.")

    encoders_path = f"/app/data/{dataset_id.replace('.csv', '')}_encoders.pkl"      # encoders_path = /app/data/123_abc_encoders.pkl

    with open(encoders_path, "wb") as f:
        pickle.dump(feature_encoders, f)

    actions.append(f"Saved feature encoders to {encoders_path}")

    # --- Step 8: High dimensionality flag ---
    n_features = len(df.columns) - 1  # exclude target
    if n_features > 25:
        flags.append(f"High dimensionality: {n_features} features after encoding. Consider PCA via reduce_dimensions tool.")

    # --- Step 9: Save outputs ---
    # --- Forecasting: sort by datetime column before saving ---
    if problem_type == "forecasting" and datetime_cols:
        sort_col = datetime_cols[0]
        df[sort_col] = df[sort_col].astype(str)  # back to string for CSV portability
        df.sort_values(sort_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        actions.append(f"Sorted by datetime column '{sort_col}' for forecasting.")
    os.makedirs("/app/data", exist_ok=True)
    cleaned_id = dataset_id.replace(".csv", "_cleaned.csv")
    cleaned_path = f"/app/data/{cleaned_id}"
    df.to_csv(cleaned_path, index=False)
    actions.append(f"Saved cleaned dataset to {cleaned_path}")
    # cleaned_path = /app/data/123_abc_cleaned.csv

    if label_map:
        label_map_path = f"/app/data/{dataset_id.replace('.csv', '')}_label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(label_map, f)
        actions.append(f"Saved label map to {label_map_path}")
        # label_map_path = /app/data/123_abc_label_map.json
    return {
        "status": "ok",
        "cleaned_dataset_id": cleaned_id,
        "original_shape": [int(df.shape[0]), int(df.shape[1])],
        "actions": actions,
        "flags": flags,
        "encoders_path": encoders_path
    }

@mcp.tool()
def check_class_balance(dataset_id: str, target_column: str) -> dict:
    """Check class distribution in a classification dataset. Flags imbalance if any class < 20%."""
    import pandas as pd

    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}. Run clean_dataset first."}

    df = pd.read_csv(path)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in dataset."}

    distribution = df[target_column].value_counts(normalize=True).sort_index()

    IMBALANCE_THRESHOLD = 0.20  # flag if any class holds less than 20% of samples

    minority_ratio = float(distribution.min())
    is_imbalanced = minority_ratio < IMBALANCE_THRESHOLD

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "target_column": target_column,
        "class_distribution": {int(k): round(float(v), 4) for k, v in distribution.items()},
        "minority_class_ratio": round(minority_ratio, 4),
        "is_imbalanced": is_imbalanced,
        "note": (
            f"Minority class holds {minority_ratio:.1%} of samples — imbalance detected. "
            f"class_weight='balanced' is already active in sklearn classifiers. "
            f"For stronger correction, consider SMOTE via imblearn."
            if is_imbalanced else
            "Class distribution looks healthy."
        ),
    }

@mcp.tool()
def hyperparameter_search(dataset_id: str, target_column: str,
                          problem_type: str, model_type: str) -> dict:
    """Run GridSearchCV for a given model type. Returns best params and CV score."""
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}. Run clean_dataset first."}

    df = pd.read_csv(path)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grids = {
        ("logistic_regression", "classification"): {"model__C": [0.01, 0.1, 1, 10], "model__max_iter": [200, 500]},
        ("linear_regression", "regression"):     {"model__fit_intercept": [True, False]},
        ("random_forest",       "classification"): {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10]},
        ("random_forest",       "regression"):     {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10]},
        ("xgboost",             "classification"): {"model__n_estimators": [50, 100], "model__learning_rate": [0.01, 0.1], "model__max_depth": [3, 6]},
        ("xgboost",             "regression"):     {"model__n_estimators": [50, 100], "model__learning_rate": [0.01, 0.1], "model__max_depth": [3, 6]},
        ("catboost",            "classification"): {"model__iterations": [50, 100], "model__learning_rate": [0.01, 0.1], "model__depth": [4, 6]},
        ("catboost",            "regression"):     {"model__iterations": [50, 100], "model__learning_rate": [0.01, 0.1], "model__depth": [4, 6]},
    }


    model_map = {
        ("logistic_regression", "classification"): LogisticRegression(random_state=42, class_weight="balanced"),
        ("linear_regression", "regression"):     LinearRegression(),
        ("random_forest",       "classification"): RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        ("random_forest",       "regression"):     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    model_key = (model_type, problem_type)
    if model_key in model_map:
        model = model_map[model_key]
    else:
        try:
            if model_type == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor
                model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0) \
                        if problem_type == "classification" \
                        else XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            elif model_type == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0) \
                        if problem_type == "classification" \
                        else CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
        except ImportError as e:
            return {"error": f"Library not installed: {e}"}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    grid_key = (model_type, problem_type)
    if grid_key not in param_grids:
        return {"error": f"No param grid defined for '{model_type}' / '{problem_type}'."}

    scoring = "f1_macro" if problem_type == "classification" else "r2"
    grid_search = GridSearchCV(pipe, param_grids[grid_key], cv=5, scoring=scoring, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = {k.replace("model__", ""): v for k, v in grid_search.best_params_.items()}
    best_score = round(float(grid_search.best_score_), 4)
    
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("automl-platform")
    with mlflow.start_run(run_name=f"gridsearch_{model_type}") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric(f"best_cv_{scoring}", best_score)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("problem_type", problem_type)
        gs_run_id = run.info.run_id

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "model_type": model_type,
        "best_params": best_params,
        "best_cv_score": best_score,
        "scoring_metric": scoring,
        "gs_run_id": gs_run_id,
        "note": "Use best_params to inform your next train_model call."
    }

@mcp.tool()
def reduce_dimensions(dataset_id: str, target_column: str) -> dict:
    """Fit PCA on train split, transform full dataset, save reduced CSV."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}. Run clean_dataset first."}

    df = pd.read_csv(path)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pca = PCA(n_components=0.95)
    pca.fit(X_train)

    X_pca = pca.transform(X)

    pca_cols = [f"pc_{i}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pca_cols)
    df_pca[target_column] = y.values

    pca_id = dataset_id.replace(".csv", "_pca.csv")
    pca_path = f"/app/data/{pca_id}"
    df_pca.to_csv(pca_path, index=False)

    import pickle
    pca_path = f"/app/data/{dataset_id.replace('.csv','')}_pca.pkl"

    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)

    return {
        "status": "ok",
        "original_dataset_id": dataset_id,
        "new_dataset_id": pca_id,
        "n_components_kept": int(pca.n_components_),
        "variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "original_n_features": X.shape[1],
        "pca_path": pca_path,
        "pca_applied": True,
        "note": f"Reduced from {X.shape[1]} features to {int(pca.n_components_)} components explaining {pca.explained_variance_ratio_.sum():.1%} variance. Use new_dataset_id for training."
    }

@mcp.tool()
def train_model(dataset_id: str, target_column: str, 
                problem_type: str, model_type: str, best_params: dict = None) -> dict:
    """Train a model on a cleaned dataset. Logs to MLflow. Returns train metrics and run_id."""
    import pandas as pd
    import pickle
    import mlflow
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score, f1_score, roc_auc_score, classification_report, mean_squared_error
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn import set_config
    set_config(enable_metadata_routing=True)

    # --- Step 1: Load cleaned dataset ---
    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Apply PCA if this dataset was reduced ---
    pca_path = f"/app/data/{dataset_id.replace('.csv','')}_pca.pkl"

    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        X = pca.transform(X)

        # Convert back to dataframe because downstream sklearn expects columns sometimes
        X = pd.DataFrame(
            X,
            columns=[f"pc_{i}" for i in range(X.shape[1])]
        )
    # --- Step 2: Train/test split ---
    # stratify=y for classification ensures class balance is preserved in both splits
    # e.g. if 80% of rows are class 0, both train and test will be ~80% class 0
    stratify = y if problem_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # --- Step 3: Select model based on type and problem ---
    model_map = {
        ("logistic_regression", "classification"): LogisticRegression(max_iter=1000,class_weight="balanced"),
        ("linear_regression",   "regression"):     LinearRegression(),
        ("random_forest",       "classification"): RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        ("random_forest",       "regression"):     RandomForestRegressor(n_estimators=100, random_state=42),
    }
    sample_weight = None
    model_key = (model_type, problem_type)
    if model_key not in model_map:
        # XGBoost and CatBoost handled separately below to keep imports optional
        try:
            if model_type == "xgboost":
                sample_weight = compute_sample_weight("balanced", y_train) \
                    if (problem_type == "classification") \
                else None
                from xgboost import XGBClassifier, XGBRegressor
                n_classes = len(y_train.unique())
                model = XGBClassifier(n_estimators=100, random_state=42, verbosity=0) \
                        if problem_type == "classification" \
                        else XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            elif model_type == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0, auto_class_weights="Balanced") \
                        if problem_type == "classification" \
                        else CatBoostRegressor(iterations=100, random_seed=42, verbose=0)
            else:
                return {"error": f"Unknown model_type '{model_type}' for problem '{problem_type}'."}
        except ImportError as e:
            return {"error": f"Library not installed: {e}"}
    else:
        model = model_map[model_key]
    if best_params:
        model.set_params(**best_params)

    # --- Step 4: Build Pipeline and fit ---
    # XGBoost sample_weight bypasses Pipeline routing limitation —
    # scaler is fit separately, then XGBoost is fit directly with weights
    if model_type == "xgboost" and problem_type == "classification":
        sample_weight = compute_sample_weight("balanced", y_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        # Wrap back into Pipeline for consistent predict/predict_proba interface
        pipe = Pipeline([("scaler", scaler), ("model", model)])
        # Replace X_test with scaled version for persistence
        X_test = X_test_scaled
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)


    # --- Step 5: Compute train metrics ---
    y_train_pred = pipe.predict(X_train)
    if problem_type == "classification":
        n_classes = len(y.unique())
        roc_auc = float(roc_auc_score(
            y_train,
            pipe.predict_proba(X_train)[:, 1] if n_classes == 2
            else pipe.predict_proba(X_train),
            **({} if n_classes == 2 else {"multi_class": "ovr"})
        ))
        train_metrics = {
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "train_f1_macro": float(f1_score(y_train, y_train_pred, average="macro")),
            "train_roc_auc": roc_auc,
            "classification_report": classification_report(y_train, y_train_pred, output_dict=True),
        }
    else:
        mse = mean_squared_error(y_train, y_train_pred)
        train_metrics = {
            "train_r2":   float(r2_score(y_train, y_train_pred)),
            "train_rmse": float(mse ** 0.5),
        }

    # --- Step 6: Persist pipeline and test split ---
    os.makedirs("/app/data", exist_ok=True)
    model_key_str = dataset_id.replace(".csv", "")
    pipeline_path = f"/app/data/{model_key_str}_{model_type}.pkl"
    test_path     = f"/app/data/{model_key_str}_test.pkl"

    with open(pipeline_path, "wb") as f:
        pickle.dump(pipe, f)
    with open(test_path, "wb") as f:
        pickle.dump((X_test, y_test), f)

    # --- Step 7: Log to MLflow ---
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("automl-platform")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type":    model_type,
            "problem_type":  problem_type,
            "dataset_id":    dataset_id,
            "train_rows":    int(X_train.shape[0]),
            "test_rows":     int(X_test.shape[0]),
            "n_features":    int(X_train.shape[1]),
        })
        # log scalar metrics only — classification_report is a nested dict
        scalar_metrics = {k: v for k, v in train_metrics.items()
                         if isinstance(v, float)}
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_param("model_path", pipeline_path)
        run_id = run.info.run_id

    return {
        "status": "ok",
        "model_type": model_type,
        "problem_type": problem_type,
        "run_id": run_id,
        "pipeline_path": pipeline_path,
        "test_path": test_path,
        "train_metrics": train_metrics,
    }

@mcp.tool()
def prepare_forecast_dataset(
    dataset_id: str,
    target_column: str,
    datetime_column: str,
    window_size: int,
    test_split: float = 0.2,
) -> dict:
    """
    Convert a cleaned forecasting CSV into sliding-window arrays for LSTM/Transformer.
    Multivariate: uses all non-datetime, non-target columns as input features.
    Saves X_train, y_train, X_test, y_test as a .npz file to /app/data/.
    Also saves _forecast_meta.json and two scalers.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import json

    # --- Block 1: Load and validate ---
    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"}

    if datetime_column not in df.columns:
        return {"error": f"Datetime column '{datetime_column}' not found. Available: {list(df.columns)}"}

    if window_size < 2:
        return {"error": "window_size must be at least 2."}

    if len(df) < window_size + 2:
        return {"error": f"Dataset too short ({len(df)} rows) for window_size={window_size}."}

    # Sort chronologically — defensive re-sort
    df = df.sort_values(datetime_column).reset_index(drop=True)

    # --- Block 2: Build feature matrix and target series ---
    # Feature columns = everything except target and datetime index
    feature_columns = [c for c in df.columns if c not in [target_column, datetime_column]]
    n_features = len(feature_columns)

    y_raw = df[target_column].values.astype(float)   # shape: (N,)

    if n_features == 0:
        # Univariate: no covariates — use lagged target as the sole input feature
        # Reshape to (N, 1) so windowing produces (samples, window_size, 1)
        # This keeps the rest of the pipeline (scalers, model input shape) identical
        print("[DEBUG] No covariate columns found — falling back to univariate mode.")
        X_raw = y_raw.reshape(-1, 1)   # shape: (N, 1)
        feature_columns = [target_column]
        n_features = 1
    else:
        X_raw = df[feature_columns].values.astype(float)   # shape: (N, n_features)

    N = len(df)

    # --- Block 3: Sliding window construction ---
    # Each sample: X[i] = rows i to i+window_size of X_raw (shape: window_size, n_features)
    #              y[i] = y_raw[i + window_size]  (the next target value)
    X_wins, y_wins = [], []
    for i in range(N - window_size):
        X_wins.append(X_raw[i : i + window_size, :])   # (window_size, n_features)
        y_wins.append(y_raw[i + window_size])

    X_wins = np.array(X_wins)   # (N - window_size, window_size, n_features)
    y_wins = np.array(y_wins)   # (N - window_size,)
    print(f"[DEBUG] df shape: {df.shape}")
    print(f"[DEBUG] N: {N}, window_size: {window_size}")
    print(f"[DEBUG] n_features: {n_features}, feature_columns: {feature_columns}")
    print(f"[DEBUG] X_wins shape: {X_wins.shape}")
    print(f"[DEBUG] y_wins shape: {y_wins.shape}")
    #print(f"[DEBUG] split_idx: {split_idx}")

    # --- Block 4: Chronological train/test split BEFORE scaling ---
    # We split first, then fit scalers on train only — avoids leakage
    split_idx = int(len(X_wins) * (1 - test_split))

    X_train_raw = X_wins[:split_idx]    # (n_train, window_size, n_features)
    X_test_raw  = X_wins[split_idx:]
    y_train_raw = y_wins[:split_idx]
    y_test_raw  = y_wins[split_idx:]

    # --- Block 5: Fit and apply scalers ---
    # Feature scaler: fit on flattened train windows, reshape back after
    # We flatten to 2D for the scaler, then restore the 3D shape
    feature_scaler = StandardScaler()
    n_train, ws, nf = X_train_raw.shape
    feature_scaler.fit(X_train_raw.reshape(-1, nf))    # fit on (n_train * window_size, n_features)

    X_train = feature_scaler.transform(X_train_raw.reshape(-1, nf)).reshape(X_train_raw.shape)
    X_test  = feature_scaler.transform(X_test_raw.reshape(-1, nf)).reshape(X_test_raw.shape)

    # Target scaler: fit on train targets only
    target_scaler = MinMaxScaler()
    target_scaler.fit(y_train_raw.reshape(-1, 1))

    y_train = target_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test  = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    # --- Block 6: Save scalers ---
    raw_stem = dataset_id.replace("_cleaned.csv", "")   # e.g. "abc123_myfile"

    feature_scaler_path = f"/app/data/{raw_stem}_forecast_feature_scaler.pkl"
    target_scaler_path  = f"/app/data/{raw_stem}_forecast_scaler.pkl"

    with open(feature_scaler_path, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(target_scaler_path, "wb") as f:
        pickle.dump(target_scaler, f)

    # --- Block 7: Save meta JSON ---
    meta = {
        "window_size":      window_size,
        "target_column":    target_column,
        "feature_columns":  feature_columns,
        "n_features":       n_features,
    }
    meta_path = f"/app/data/{raw_stem}_forecast_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # --- Block 8: Save windowed arrays ---
    npz_id   = dataset_id.replace("_cleaned.csv", "_windows.npz")
    npz_path = f"/app/data/{npz_id}"
    np.savez(npz_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    return {
        "status":           "ok",
        "npz_id":           npz_id,
        "window_size":      window_size,
        "n_features":       n_features,
        "feature_columns":  feature_columns,
        "n_train_samples":  int(len(X_train)),
        "n_test_samples":   int(len(X_test)),
        "input_shape":      list(X_train.shape),   # (samples, window_size, n_features)
        "target_column":    target_column,
        "scaler_path":      target_scaler_path,
        "feature_scaler_path": feature_scaler_path,
        "meta_path":        meta_path,
    }


@mcp.tool()
def train_ffn(
    dataset_id: str,
    target_column: str,
    problem_type: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Train a feedforward neural network on a cleaned tabular CSV.
    Supports classification and regression.
    Saves model as .keras and test split as .pkl to /app/data/.
    """
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import mlflow
    import tensorflow as tf

    # --- Reproducibility seeds ---
    import random
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Block 1: Load, split, scale ---
    path = f"/app/data/{dataset_id}"
    if not os.path.exists(path):
        return {"error": f"Dataset not found at {path}"}

    df = pd.read_csv(path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    if problem_type not in ("classification", "regression"):
        return {"error": f"problem_type must be 'classification' or 'regression', got '{problem_type}'."}

    X = df.drop(columns=[target_column]).values.astype(float)
    y = df[target_column].values

    # --- Apply PCA if this dataset was reduced ---
    pca_path = f"/app/data/{dataset_id.replace('.csv','')}_pca.pkl"

    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)

        X = pca.transform(X)

    # FFN expects numpy arrays
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale — fit on train only, transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    scaler_path = f"/app/data/{dataset_id.replace('.csv', '')}_ffn_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Persist test split for evaluate_model later
    test_split_path = f"/app/data/{dataset_id.replace('.csv', '')}_ffn_test.pkl"
    with open(test_split_path, "wb") as f:
        pickle.dump({"X_test": X_test, "y_test": y_test}, f)
    
    # --- Block 2: Build model ---
    n_features = X_train.shape[1]

    if problem_type == "classification":
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            output_layer = tf.keras.layers.Dense(1, activation="sigmoid")
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            output_layer = tf.keras.layers.Dense(n_classes, activation="softmax")
            loss = "sparse_categorical_crossentropy"
            metrics = ["accuracy"]
    else:
        output_layer = tf.keras.layers.Dense(1, activation="linear")
        loss = "mse"
        metrics = ["mae"]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        output_layer,
    ])

    model.compile(optimizer="adam", loss=loss, metrics=metrics)

    # --- Block 3: Train + MLflow + save ---
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("automl-ffn")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "dataset_id": dataset_id,
            "target_column": target_column,
            "problem_type": problem_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_features": n_features,
            "architecture": "128-64-32-ffn",
        })

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Log final epoch metrics only — full history would bloat MLflow
        final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final_metrics)

        # Save model
        model_id = dataset_id.replace(".csv", "_ffn.keras")
        model_path = f"/app/data/{model_id}"
        model.save(model_path)

        run_id = run.info.run_id

    return {
        "status": "ok",
        "model_id": model_id,
        "run_id": run_id,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "final_metrics": final_metrics,
        "test_split_path": test_split_path,
    }

@mcp.tool()
def evaluate_ffn(
    dataset_id: str,
    target_column: str,
    problem_type: str,
    run_id: str,
) -> dict:
    import pickle
    import numpy as np
    import tensorflow as tf

    # --- Block 1: Load test split and model ---
    test_split_path = f"/app/data/{dataset_id.replace('.csv', '')}_ffn_test.pkl"
    model_path = f"/app/data/{dataset_id.replace('.csv', '_ffn.keras')}"

    if not os.path.exists(test_split_path):
        return {"error": f"Test split not found at {test_split_path}"}
    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}"}

    with open(test_split_path, "rb") as f:
        test_data = pickle.load(f)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    model = tf.keras.models.load_model(model_path)

    # --- Block 2: Predict and compute metrics ---
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        r2_score, mean_squared_error
    )

    raw_preds = model.predict(X_test)
    output_shape = raw_preds.shape

    if problem_type == "classification":
        if output_shape[1] == 1:
            # Binary — sigmoid output
            y_pred = (raw_preds > 0.5).astype(int).flatten()
            y_prob = raw_preds.flatten()
        else:
            # Multiclass — softmax output
            y_pred = np.argmax(raw_preds, axis=1)
            y_prob = raw_preds  # shape (n, n_classes) for roc_auc

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        try:
            if output_shape[1] == 1:
                roc = roc_auc_score(y_test, y_prob)
            else:
                roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc = None

        metrics = {"accuracy": acc, "f1_macro": f1}
        if roc is not None:
            metrics["roc_auc"] = roc

    else:
        # Regression
        y_pred = raw_preds.flatten()
        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics = {"r2": r2, "rmse": rmse}
    
    # --- Block 3: Log to MLflow and return ---
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("automl-ffn")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "problem_type": problem_type,
        "run_id": run_id,
        "metrics": metrics,
    }

@mcp.tool()
def train_lstm(
    npz_id: str,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Train a stacked LSTM for multivariate time series forecasting.
    Loads windowed arrays from prepare_forecast_dataset output (.npz).
    Saves model as .keras and logs to MLflow.
    """
    import random
    import numpy as np
    import tensorflow as tf
    import mlflow

    # --- Reproducibility seeds ---
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Block 1: Load .npz ---
    npz_path = f"/app/data/{npz_id}"
    if not os.path.exists(npz_path):
        return {"error": f".npz file not found at {npz_path}"}

    data = np.load(npz_path)
    X_train = data["X_train"]   # shape: (samples, window_size, 1)
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    window_size = X_train.shape[1]
    n_features  = X_train.shape[2]

    # --- Block 2: Build LSTM model ---
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1, activation="linear"),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # --- Block 3: Train + MLflow + save ---
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("automl-lstm")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "npz_id": npz_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
            "n_features": n_features,
            "architecture": "128-64-lstm-64-32-dense",
        })

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Log final epoch metrics only — full history would bloat MLflow
        final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final_metrics)

        # Save model
        model_id = npz_id.replace("_windows.npz", "_lstm.keras")
        model_path = f"/app/data/{model_id}"
        model.save(model_path)

        run_id = run.info.run_id

    return {
        "status": "ok",
        "model_id": model_id,
        "run_id": run_id,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "final_metrics": final_metrics,
    }


# ── Transformer custom layers ────────────────────────────────────────────────

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

def build_model_t(time_step, n_features=1 , embed_dim=64, num_heads=4, ff_dim=128,
                  num_layers=2, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(time_step, n_features))  # multivariate support 
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    x = PositionalEncoding(time_step, embed_dim)(x)
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout_rate)
    x = encoder(x)
    x = x[:, -1, :]                          # take last timestep only
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)

@mcp.tool()
def train_transformer(
    npz_id: str,
    embed_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout_rate: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
) -> dict:
    """
    Train a Transformer encoder for multivariate time series forecasting.
    Loads windowed arrays from prepare_forecast_dataset output (.npz).
    Saves model as .keras and logs to MLflow.
    """
    import random
    import tensorflow as tf
    import mlflow

    # --- Reproducibility seeds ---
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # --- Block 2: Load .npz ---
    npz_path = f"/app/data/{npz_id}"
    if not os.path.exists(npz_path):
        return {"error": f".npz file not found at {npz_path}"}

    data = np.load(npz_path)
    X_train = data["X_train"]   # shape: (samples, window_size, 1)
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    # --- Block 3: Build + compile + train + save ---
    model = build_model_t(
        time_step=window_size,
        n_features = n_features,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("automl-transformer")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "npz_id": npz_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
            "n_features": n_features,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "architecture": "transformer-encoder",
        })

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        )

        final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final_metrics)

        model_id = npz_id.replace("_windows.npz", "_transformer.keras")
        model_path = f"/app/data/{model_id}"
        model.save(model_path)

        run_id = run.info.run_id

    return {
        "status": "ok",
        "model_id": model_id,
        "run_id": run_id,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "final_metrics": final_metrics,
    }


"""
What `evaluate_model` does:
* Load the fitted pipeline from `/app/data/{dataset_id}_{model_type}.pkl`
* Load the test split from `/app/data/{dataset_id}_test.pkl`
* Compute test metrics (same set as train, but on held-out data)
* Log metrics to the same MLflow run using the `run_id` from `train_model
* Return metrics dict
"""
@mcp.tool()
def evaluate_model(dataset_id: str, model_type: str,
                   problem_type: str, run_id: str) -> dict:
    """Evaluate a trained model on the held-out test split. Logs to existing MLflow run."""
    import pickle
    import mlflow
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        classification_report, r2_score, mean_squared_error
    )

    # --- Step 1: Load pipeline and test split ---
    model_key_str = dataset_id.replace(".csv", "")
    pipeline_path = f"/app/data/{model_key_str}_{model_type}.pkl"
    test_path     = f"/app/data/{model_key_str}_test.pkl"

    for p in [pipeline_path, test_path]:
        if not os.path.exists(p):
            return {"error": f"File not found: {p}. Run train_model first."}

    with open(pipeline_path, "rb") as f:
        pipe = pickle.load(f)
    with open(test_path, "rb") as f:
        X_test, y_test = pickle.load(f)

    # --- Step 2: Compute test metrics ---
    y_pred = pipe.predict(X_test)

    if problem_type == "classification":
        n_classes = len(y_test.unique())
        roc_auc = float(roc_auc_score(
            y_test,
            pipe.predict_proba(X_test)[:, 1] if n_classes == 2
            else pipe.predict_proba(X_test),
            **({} if n_classes == 2 else {"multi_class": "ovr"})
        ))
        test_metrics = {
            "test_accuracy":  float(accuracy_score(y_test, y_pred)),
            "test_f1_macro":  float(f1_score(y_test, y_pred, average="macro")),
            "test_roc_auc":   roc_auc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        test_metrics = {
            "test_r2":   float(r2_score(y_test, y_pred)),
            "test_rmse": float(mse ** 0.5),
        }

    # --- Step 3: Log to existing MLflow run ---
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("automl-platform")
    with mlflow.start_run(run_id=run_id):
        scalar_metrics = {k: v for k, v in test_metrics.items()
                         if isinstance(v, float)}
        mlflow.log_metrics(scalar_metrics)

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "model_type": model_type,
        "run_id": run_id,
        "test_metrics": test_metrics,
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
