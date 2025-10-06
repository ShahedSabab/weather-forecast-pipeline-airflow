from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from mlflow_provider.hooks.client import MLflowClientHook
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from airflow.hooks.base import BaseHook
import os 
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow

# Config
POSTGRES_CONN_ID = "postgres_default"
MLFLOW_CONN_ID = "mlflow_default"
MINIO_CONN_ID = "minio_conn"
TABLE_NAME = "weather_data"
TEST_DAYS = 3
FORECAST_DAYS = 3
EXPERIMENT_NAME = "Weather_Forecasting_v2"
ARTIFACT_BUCKET = "s3://mlflow-artifacts"
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"

def setup_minio_env():
    """Setup MinIO environment variables - call this in every task that uses MLflow"""
    conn = BaseHook.get_connection(MINIO_CONN_ID)
    os.environ["AWS_ACCESS_KEY_ID"] = conn.login
    os.environ["AWS_SECRET_ACCESS_KEY"] = conn.password
    # MinIO endpoint - use extra_dejson if host doesn't include port
    minio_host = conn.extra_dejson.get('host') if conn.extra_dejson.get('host') else conn.host
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_host
    print(f"✓ MinIO configured: {minio_host}")

default_args = {
    "owner": "shahed_sabab",
    "start_date": datetime.utcnow() - timedelta(days=1)
}

with DAG(
    dag_id="forecast_weather_pipeline",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["forecast", "mlflow", "prophet"],
) as dag:

    # ==================================================
    # 1️ Read Weather Data from Postgres
    # ==================================================
    @task
    def read_weather_data():
        pg = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg.get_conn()
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY date", conn)
        conn.close()
        return df.to_json(orient="records")

    # ==================================================
    # 2 Createrate MLflow Experiment
    # ==================================================
    @task
    def create_experiment():
        """Create MLflow experiment using the MLflow provider hook"""
        setup_minio_env()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print(f"Creating new experiment: {EXPERIMENT_NAME}")
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=ARTIFACT_BUCKET
            )
            print(f"✓ Created experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        
        return EXPERIMENT_NAME

    # ==================================================
    # 3 Train and log model to MLflow
    # ==================================================
    @task
    def train_and_log_model(df_json: str, column_name: str, metric_prefix: str, experiment_name: str):
        """Train model and log to MLflow using mlflow Python package"""
        setup_minio_env()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=experiment_name)
        
        # Prepare data
        df = pd.read_json(df_json)
        df = df.rename(columns={"date": "ds", column_name: "y"})
        df["ds"] = pd.to_datetime(df["ds"])
        
        train = df.iloc[:-TEST_DAYS]
        test = df.iloc[-TEST_DAYS:]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"{metric_prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log parameters
            mlflow.log_param("model_type", "Prophet")
            mlflow.log_param("metric_name", column_name)
            mlflow.log_param("test_days", TEST_DAYS)
            mlflow.log_param("forecast_days", FORECAST_DAYS)
            mlflow.log_param("train_size", len(train))
            mlflow.log_param("test_size", len(test))
            
            # Train model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(train)
            
            # Make predictions
            future = model.make_future_dataframe(periods=TEST_DAYS + FORECAST_DAYS, freq="D")
            forecast = model.predict(future)
            
            # Calculate metrics
            y_true = test["y"].values
            y_pred = forecast.iloc[-(TEST_DAYS + FORECAST_DAYS):-FORECAST_DAYS]["yhat"].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mape", mape)
            
            # Log model
            try:
                mlflow.prophet.log_model(model, artifact_path="model")
                print(f"✓ Model logged successfully to MinIO")
            except Exception as e:
                print(f"⚠ WARNING: Could not log model: {e}")
                print(f"   Metrics have been logged successfully. Continuing...")
            
            print(f"{metric_prefix} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            run_id = mlflow.active_run().info.run_id
            
            # Prepare forecast
            forecast_future = forecast[["ds", "yhat"]].tail(30).copy()
            forecast_future["ds"] = forecast_future["ds"].dt.strftime("%Y-%m-%d")
            
            return {
                "forecast": forecast_future.to_dict(orient="records"),
                "metrics": {"mae": mae, "rmse": rmse, "mape": mape},
                "run_id": run_id
            }

    
    # ==================================================
    # 4 Combine forecasts
    # ==================================================

    @task
    def combine_forecasts(max_result, min_result, mean_result):
        """Combine all forecasts"""
        forecast_max = pd.DataFrame(max_result["forecast"])
        forecast_min = pd.DataFrame(min_result["forecast"])
        forecast_mean = pd.DataFrame(mean_result["forecast"])
        
        forecast_combined = (
            forecast_max
            .merge(forecast_min, on='ds', suffixes=('_max', '_min'))
            .merge(forecast_mean, on='ds')
        )
        
        forecast_combined = forecast_combined.rename(columns={
            'yhat_max': 'temperature_max',
            'yhat_min': 'temperature_min',
            'yhat': 'temperature_mean'
        })
        
        print("Combined Forecast:")
        print(forecast_combined)
        
        return forecast_combined.to_dict(orient="records")
    
    # ==================================================
    # 5 Report: Plot combined forecast and log to MLflow
    # ==================================================
    @task
    def plot_combined_forecast(combined_df: list):
        """
        Create a line plot for max, min, mean temperatures.
        Last 3 points are forecasts and will be highlighted.
        Log the plot to MLflow.
        """
        setup_minio_env()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
        
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(combined_df)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Split historical vs forecast
        forecast_days = 3
        df_hist = df.iloc[:-forecast_days]
        df_forecast = df.iloc[-forecast_days:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical lines
        ax.plot(df_hist['ds'], df_hist['temperature_max'], label='Max Temp (Historical)', color='red')
        ax.plot(df_hist['ds'], df_hist['temperature_min'], label='Min Temp (Historical)', color='blue')
        ax.plot(df_hist['ds'], df_hist['temperature_mean'], label='Mean Temp (Historical)', color='green')

        # Plot forecast lines + dots
        ax.plot(
            df_forecast['ds'], df_forecast['temperature_max'],
            color='red', linestyle=':', marker='o', label='Max Temp (Forecast)'
        )
        ax.plot(
            df_forecast['ds'], df_forecast['temperature_min'],
            color='blue', linestyle=':', marker='o', label='Min Temp (Forecast)'
        )
        ax.plot(
            df_forecast['ds'], df_forecast['temperature_mean'],
            color='green', linestyle=':', marker='o', label='Mean Temp (Forecast)'
        )

        # Chart settings
        ax.set_title("Weather Forecast vs Historical")
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature")
        ax.grid(True)
        ax.legend()

        # Log figure to MLflow
        with mlflow.start_run(run_name=f"combined_forecast_plot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_figure(fig, "forecast_plots/combined_forecast.png")
            print("✅ Combined forecast plot successfully logged to MLflow/MinIO")

        plt.close(fig)
        return True

    # DAG Flow
    data = read_weather_data()
    exp_id = create_experiment()
    
    max_result = train_and_log_model(data, "temperatures_max", "max_temp", EXPERIMENT_NAME)
    min_result = train_and_log_model(data, "temperatures_min", "min_temp", EXPERIMENT_NAME)
    mean_result = train_and_log_model(data, "temperature_mean", "mean_temp", EXPERIMENT_NAME)
    
    combined = combine_forecasts(max_result, min_result, mean_result)
    forecast_plot = plot_combined_forecast(combined)

    # Set task dependencies explicitly (optional, but clearer)
    data >> exp_id
    [max_result, min_result, mean_result] >> combined >> forecast_plot