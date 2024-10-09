import psycopg2
import os
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def postgress_cloud_certs():
    # Define the target directory and file path
    target_dir = Path.home() / ".postgresql"
    file_path = target_dir / "root.crt"

    # Create the directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Check if the file does not exist
    if not file_path.exists() :
        # Download the file using wget
        url = "https://storage.yandexcloud.net/cloud-certs/CA.pem"
        result = subprocess.run(["wget", url, "--output-document", str(file_path)])

        # Check if the download was successful
        if result.returncode == 0 :
            # Set the file permissions
            file_path.chmod(0o600)
            print(f"Downloaded and set permissions for {file_path}")
        else :
            print(f"Failed to download {url}")
    else :
        print(f"File {file_path} already exists")
def get_connection(dbname="db3"):
    postgress_user = os.environ.get("POSTGRESS_USER")
    postgress_pwd = os.environ.get("POSTGRESS_PWD")
    postgress_host = os.environ.get("POSTGRESS_HOST")
    try:
        postgress_cloud_certs()
    except:
        print("postgress_cloud_certs failed")
    return psycopg2.connect(f"""
        host={postgress_host}
        port=6432
        sslmode=verify-full
        dbname={dbname}
        user={postgress_user}
        password={postgress_pwd}
        target_session_attrs=read-write
    """)


def upload_dataframe_to_postgres (df, pgc, table_name="SYNTH_BOY_GIRL") :
    """
    Uploads a DataFrame to a PostgreSQL table. If the table doesn't exist, it will be created.
    If a column doesn't exist, it will be added.

    :param df: pd.DataFrame, the DataFrame to upload
    :param postgres_connection: psycopg2 connection, the PostgreSQL connection object
    :param table_name: str, the name of the table to upload data to
    """
    try :
        # Rollback any existing transactions
        pgc.rollback()
        cur = pgc.cursor()

        # Check if table exists
        cur.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='{table_name}');")
        if not cur.fetchone()[0] :
            print(f"Table '{table_name}' does not exist, creating a new one.")
            table_boy_girl_creation(pgc,table_name)

        # Fetch existing columns in the table
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';")
        existing_columns = {row[0] for row in cur.fetchall()}

        # Add any missing columns to the table
        missing_columns = set(df.columns) - existing_columns
        for column in missing_columns :
            col_type = 'TEXT' if df[column].dtype == 'object' else 'DOUBLE PRECISION'
            alter_query = f"ALTER TABLE {table_name} ADD COLUMN {column} {col_type};"
            cur.execute(alter_query)
            print(f"Added missing column '{column}' to table '{table_name}'.")

        pgc.commit()

        # Insert data into the table
        columns = [column for column in df.columns if column != 'id']
        df=df[columns]
        for row in df.itertuples(index=False) :
            insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))});"
            cur.execute(insert_query, row)

        pgc.commit()

    except Exception as e :
        print(f"An error occurred: {e}")
        pgc.rollback()

    finally :
        cur.close()


def table_boy_girl_creation(pgc,table_name):
    cur = pgc.cursor()
    create_new_table_querry = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
      id SERIAL PRIMARY KEY,
      original TEXT,
      paraphrase TEXT,
      complexity TEXT,
      changes TEXT[],
      model TEXT,
      cost FLOAT,
      date TIMESTAMP
    );"""
    cur.execute(create_new_table_querry)
    pgc.commit()

def download_table_as_dataframe (postgres_connection, table_name="SYNTH_BOY_GIRL") :
    """
    Downloads the contents of a specified table and returns it as a pandas DataFrame.

    :param table_name: str, the name of the table to download
    :return: pd.DataFrame, the contents of the table as a DataFrame
    """
    postgres_connection.rollback()
    cur = postgres_connection.cursor()
    # Query to select all data from the table
    query = f"SELECT * FROM {table_name};"
    cur.execute(query)

    # Fetch all data from the table
    rows = cur.fetchall()

    # Get column names
    colnames = [desc[0] for desc in cur.description]

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=colnames)

    cur.close()
    return df