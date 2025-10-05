from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from mlflow_provider.hooks.client import MLflowClientHook
from datetime import datetime, timedelta
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import uuid

# Config
POSTGRES_CONN_ID = "postgres_default"
MLFLOW_CONN_ID = "mlflow_default"
TABLE_NAME = "weather_data"
TEST_DAYS = 3
FORECAST_DAYS = 3
EXPERIMENT_NAME = "Weather_Forecasting_Prophet"

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
    # 1️⃣ Read Weather Data from Postgres
    # ==================================================
    @task
    def read_weather_data():
        pg = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg.get_conn()
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY date", conn)
        conn.close()
        return df.to_json(orient="records")  # XCom safe

    # =========================
    # Forecast Function
    # =========================
    def forecast_and_log(df_json, column_name, metric_prefix):
        df = pd.read_json(df_json)
        df = df.rename(columns={"date": "ds", column_name: "y"})
        df["ds"] = pd.to_datetime(df["ds"])

        train = df.iloc[:-TEST_DAYS]
        test = df.iloc[-TEST_DAYS:]

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(train)

        future = model.make_future_dataframe(periods=TEST_DAYS + FORECAST_DAYS, freq="D")
        forecast = model.predict(future)

        y_true = test["y"].values
        y_pred = forecast.iloc[-(TEST_DAYS + FORECAST_DAYS):-FORECAST_DAYS]["yhat"].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"{metric_prefix} - MAE: {mae}, RMSE: {rmse}")
        # # MLflow logging
        # mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
        # run_name = f"{metric_prefix}_{datetime.utcnow().isoformat()}"
        
        # # Create or get experiment
        # mlflow_hook.run(
        #     endpoint="api/2.0/mlflow/experiments/get-by-name",
        #     request_params={"experiment_name": EXPERIMENT_NAME},
        # )

        # # Create a run
        # run_resp = mlflow_hook.run(
        #     endpoint="api/2.0/mlflow/runs/create",
        #     request_params={"experiment_id": "0", "run_name": run_name}
        # )
        # run_id = run_resp["run"]["info"]["run_id"]

        # # Log metrics
        # mlflow_hook.run(
        #     endpoint="api/2.0/mlflow/runs/log-metric",
        #     request_params={"run_id": run_id, "key": f"{metric_prefix}_mae", "value": mae}
        # )
        # mlflow_hook.run(
        #     endpoint="api/2.0/mlflow/runs/log-metric",
        #     request_params={"run_id": run_id, "key": f"{metric_prefix}_rmse", "value": rmse}
        # )

        forecast_future = forecast[["ds", "yhat"]].tail(FORECAST_DAYS).copy()
        forecast_future["ds"] = forecast_future["ds"].dt.strftime("%Y-%m-%d")
        return forecast_future.to_dict(orient="records")

    # =========================
    # Forecast Tasks
    # =========================
    @task
    def train_max_temp(json_df: str):
        return forecast_and_log(json_df, "temperatures_max", "max_temp")

    @task
    def train_min_temp(json_df: str):
        return forecast_and_log(json_df, "temperatures_min", "min_temp")

    @task
    def train_mean_temp(json_df: str):
        return forecast_and_log(json_df, "temperature_mean", "mean_temp")
    # =========================
    # 5️⃣ Combine Forecasts Task
    # =========================
    @task
    def forecast_df(forecast_max_json, forecast_min_json, forecast_mean_json):
        import pandas as pd

        # Convert XCom JSON back to DataFrames
        forecast_max = pd.DataFrame(forecast_max_json)
        forecast_min = pd.DataFrame(forecast_min_json)
        forecast_mean = pd.DataFrame(forecast_mean_json)

        # Merge forecasts on 'ds'
        forecast_combined = (
            forecast_max
            .merge(forecast_min, on='ds')
            .merge(forecast_mean, on='ds')
        )

        # Rename columns for clarity
        forecast_combined = forecast_combined.rename(columns={
            'yhat_x': 'temperature_max',
            'yhat_y': 'temperature_min',
            'yhat': 'temperature_mean'
        })

        print("Combined Forecast:")
        print(forecast_combined)

        # Return as XCom-safe JSON
        return forecast_combined.to_dict(orient="records")

    # =========================
    # DAG Flow
    # =========================

    data_json = read_weather_data()
    max_task = train_max_temp(data_json)
    min_task = train_min_temp(data_json)
    mean_task = train_mean_temp(data_json)

    combined_task = forecast_df(max_task, min_task, mean_task)

    data_json >> [max_task, min_task, mean_task] >> combined_task