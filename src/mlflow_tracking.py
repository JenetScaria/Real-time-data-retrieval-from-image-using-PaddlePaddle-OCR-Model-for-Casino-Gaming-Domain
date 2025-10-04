import mlflow
import mlflow.paddle
import yaml
from datetime import datetime
import logging

class MLflowTracker:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.load_config(config_path)
        self.setup_mlflow()
    
    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def setup_mlflow(self):
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def start_training_run(self, run_name: str):
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(self.config['model'])
    
    def log_training_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path: str):
        mlflow.paddle.log_model(model, artifact_path)
    
    def log_ocr_performance(self, accuracy: float, processing_time: float, frames_processed: int):
        mlflow.log_metric("ocr_accuracy", accuracy)
        mlflow.log_metric("avg_processing_time", processing_time)
        mlflow.log_metric("frames_processed", frames_processed)