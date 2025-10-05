from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from datetime import datetime, timedelta
import pandas as pd
import requests
import json

LATITUDE = 53.5501
LONGITUDE = -113.4687
START_DATE = '2025-09-01'
END_DATE = '2025-10-02'
POSTGRES_CONN_ID = "postgres_default"
API_CONN_ID = "open_meteo"


default_args={
    'owner':'shahed_sabab',
    'start_date': datetime.utcnow() - timedelta(days=1)
}

## DAG DEFINITION

with DAG(dag_id="weather_etl_pipeline", default_args=default_args, schedule="@daily", catchup=False) as dag:

    # ==================================================
    # 1️⃣ Extract Weather Data from Open-Meteo API
    # ==================================================
    @task
    def extract_weather_data():
        """
        Extract daily weather data from the Open-Meteo Archive API.
        Returns JSON response with daily fields:
        wind_speed_10m_mean, relative_humidity_2m_mean, temperature_2m_mean, daylight_duration
        """
        endpoint_ = (
            f"v1/archive?"
            f"latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&start_date={START_DATE}&end_date={END_DATE}"
            f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean"
            f"&timezone=auto"
        )

        http = HttpHook(method="GET", http_conn_id=API_CONN_ID)
        response = http.run(endpoint=endpoint_)

        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        return response.json()

    # ==================================================
    # 2️⃣ Transform Weather Data
    # ==================================================
    @task
    def transform_weather_data(weather_data: dict) -> list[tuple]:
        """
        Transform the extracted daily weather data into a list of tuples.
        Each tuple is structured as:
        (latitude, longitude, date, temperature_2m_max, temperature_2m_min, temperature_2m_mean)
        """
        latitude = weather_data.get("latitude")
        longitude = weather_data.get("longitude")
        daily_data = weather_data.get("daily", {})

        dates = daily_data.get("time", [])
        temperatures_max = daily_data.get("temperature_2m_max", [])
        temperatures_min = daily_data.get("temperature_2m_min", [])
        temperatures_mean = daily_data.get("temperature_2m_mean", [])

        transformed_records = []
        for i in range(len(dates)):
            record = (
                latitude,
                longitude,
                dates[i],
                temperatures_max[i] if i < len(temperatures_max) else None,
                temperatures_min[i] if i < len(temperatures_min) else None,
                temperatures_mean[i] if i < len(temperatures_mean) else None,
            )
            transformed_records.append(record)

        return transformed_records
    

    # ==================================================
    # 3️⃣ Load weather data to a datastore
    # ==================================================
    @task
    def load_weather_data_to_db(weather_tuples: list[tuple]) -> None:
        """
        Load transformed daily weather data into PostgreSQL.
        """
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # Drop table if it exists
        cursor.execute("DROP TABLE IF EXISTS weather_data_mini;")

        # Create table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data_mini (
            latitude FLOAT,
            longitude FLOAT,
            date DATE,
            temperatures_max FLOAT,
            temperatures_min FLOAT,
            temperature_mean FLOAT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Adjust insert query to match new column order
        insert_query = """
        INSERT INTO weather_data_mini (
            latitude, longitude, date, temperatures_max, temperatures_min, temperature_mean
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.executemany(insert_query, weather_tuples)

        conn.commit()
        cursor.close()
        conn.close()

    # ==================================================
    # 4️⃣ Test: Validation of data in datastore
    # ==================================================
    @task
    def validate_data_from_db() -> str:
        """
        Read weather data from PostgreSQL, return top 5 rows as a string for XCom.
        """
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()

        query = "SELECT * FROM weather_data_mini;" 
        df = pd.read_sql(query, conn)

        conn.close()

        # Convert DataFrame to JSON
        df_json = df.head(5).to_json(orient="records")  # List of dicts

        expected_columns = [
            "latitude", "longitude", "date",
            "temperatures_max", "temperatures_min", "temperature_mean"
        ]

        missing = [c for c in expected_columns if c not in df.columns]

        print("Total number of rows:", len(df))
        print("Number of missing values:\n", df.isnull().sum())
        print("Missing columns: " + str(missing) if missing else "None")

        return df_json  # Will be stored in XCom


    # Task pipeline
    extract_task = extract_weather_data()
    transform_task = transform_weather_data(extract_task)
    load_task = load_weather_data_to_db(transform_task)
    read_task = validate_data_from_db()

    # Chain tasks
    extract_task >> transform_task >> load_task >> read_task