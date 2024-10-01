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


def table_boy_girl_creation():
    table_name = "SYNTH_BOY_GIRL"
    cur = postgres_connection.cursor()
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
    postgres_connection.commit()

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