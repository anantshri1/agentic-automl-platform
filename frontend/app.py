import gradio as gr
import requests
import os
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

def upload_and_run(file, target_column):
    if file is None:
        return "Please upload a CSV file.", "", "", "", gr.update(), gr.update(), gr.update(), gr.update()

    # Step 1: Upload file
    with open(file.name, "rb") as f:
        upload_resp = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": f}
        )
    upload_data = upload_resp.json()
    job_id = upload_data["job_id"]
    filename = upload_data["filename"]
    original_filename = file.name.split("/")[-1]

    # Step 2: Run workflow
    run_resp = requests.post(
        f"{BACKEND_URL}/run",
        json={
            "job_id": job_id,
            "filename": original_filename,
            "target_column": target_column,
            "problem_type": ""   # backend detects this — placeholder satisfies Pydantic
        }
    )
    run_data = run_resp.json()
    print("full run_data:", run_data)
    print("full run_data:", run_data)
    print("problem_type from run_data:", run_data.get("problem_type", "NOT FOUND"))
    agent_response = run_data["results"]["agent_response"]
    train_filename = run_data["filename"]
    problem_type = run_data.get("problem_type", "")

    # Determine UI visibility based on problem type
    # Forecasting: hide file upload + CSV output, show image output
    # Everything else: show file upload + CSV output, hide image output
    is_forecast = problem_type == "forecasting"
    return (
        agent_response,
        job_id,
        train_filename,
        problem_type,
        gr.update(visible=not is_forecast),   # test_file_input
        gr.update(visible=not is_forecast),   # predictions_output
        gr.update(visible=is_forecast),       # forecast_image_output
        gr.update(                            # model_dropdown: swap choices
            choices=["lstm", "transformer"] if is_forecast
                    else ["catboost", "xgboost", "random_forest", "linear_regression", "logistic_regression", "ffn"],
            value=None
        ),
    )

def predict(job_id, train_filename, problem_type, model_type, file):
    if not job_id or not train_filename:
        return "Please run a training job first (Tab 1).", None, None

    if problem_type == "forecasting":
        # No test file upload needed — backend loads X_test from the saved .npz
        predict_resp = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "job_id": job_id,
                "train_filename": train_filename,
                "test_filename": "",        # unused by forecasting branch in routes.py
                "model_type": model_type,
                "problem_type": problem_type,
            }
        )
        if predict_resp.status_code != 200:
            return f"Prediction failed: {predict_resp.text}", None, None

        # Save PNG to /tmp and return path for gr.Image
        plot_path = "/tmp/forecast_plot.png"
        with open(plot_path, "wb") as f:
            f.write(predict_resp.content)

        return "Forecast plot ready.", None, plot_path   # status, no CSV, image path

    else:
        # Non-forecasting: require test file upload
        if file is None:
            return "Please upload a test CSV file.", None, None

        # Step 1: Upload test CSV
        with open(file.name, "rb") as f:
            upload_resp = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": f}
            )
        upload_data = upload_resp.json()
        print("upload response:", upload_data)
        test_filename = upload_data["filename"]

        # Step 2: Run predict
        predict_resp = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "job_id": job_id,
                "train_filename": train_filename,
                "test_filename": test_filename,
                "model_type": model_type,
                "problem_type": problem_type,
            }
        )

        # Step 3: Save predictions CSV
        output_path = "/tmp/predictions.csv"
        with open(output_path, "wb") as f:
            f.write(predict_resp.content)

        return "Predictions ready.", output_path, None   # status, CSV path, no image
    

with gr.Blocks(title="AutoML Platform") as demo:
    # Hidden state — persists across tabs
    job_id_state = gr.State("")
    train_filename_state = gr.State("")
    problem_type_state = gr.State("")

    with gr.Tab("Train"):
        gr.Markdown("## Upload Dataset & Run AutoML")
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        target_input = gr.Textbox(label="Target Column")
        run_btn = gr.Button("Run")
        result_output = gr.Markdown(label="Agent Response")
        # run_btn.click registered below, after Predict tab components exist

    with gr.Tab("Predict"):
        gr.Markdown("## Run Predictions on New Data")
        model_dropdown = gr.Dropdown(
            choices=["catboost", "xgboost", "random_forest", "linear_regression", "logistic_regression", "ffn"],
            label="Model Type"
        )
        test_file_input = gr.File(label="Upload Test CSV", file_types=[".csv"], visible=True)
        predict_btn = gr.Button("Predict")
        predict_status = gr.Textbox(label="Status")
        predictions_output = gr.File(label="Download Predictions", visible=True)
        forecast_image_output = gr.Image(label="Forecast Plot", visible=False)

        predict_btn.click(
            fn=predict,
            inputs=[job_id_state, train_filename_state, problem_type_state, model_dropdown, test_file_input],
            outputs=[predict_status, predictions_output, forecast_image_output]
        )

    # Registered here so all output components are already defined
    run_btn.click(
        fn=upload_and_run,
        inputs=[file_input, target_input],
        outputs=[
            result_output,
            job_id_state,
            train_filename_state,
            problem_type_state,
            test_file_input,
            predictions_output,
            forecast_image_output,
            model_dropdown,
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
