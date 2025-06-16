import mlflow
from typing import Optional
from mlflow.models import infer_signature
import mlflow.sklearn

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_model(
        self,
        model,
        X,
        y,
        run_name: Optional[str] = None,
        metrics: Optional[dict] = None,
        parameters: Optional[dict] = None,
        artifact_paths: Optional[list] = None
    ):
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            if parameters:
                mlflow.log_params(parameters)

            # Log metrics
            mlflow.log_metric("accuracy", model.score(X, y))
            if metrics:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)

            # Log model
            signature = infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(model, "model", signature=signature)  # type: ignore

            # Log any additional artifacts
            if artifact_paths:
                for path in artifact_paths:
                    mlflow.log_artifact(path)

            run_id = run.info.run_id
            print(f"Model logged under run: {run_id}")
            return run_id
