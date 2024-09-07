# script .. load_data.py
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

#load environment varibles from .env file
load_dotenv()

# Fetch databse connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def load_data_from_postgres(query):
    """
    Connects to the PostgreSQL database and loads dataon the provided SQL queryy.
    
    :param query: SQL query to execute.
    :return: DataFrame containing the results of the query.
    """
    try:
        # establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        # Load the data using pandas
        df = pd.read_sql_query(query, connection)
        
        # close the database connection
        connection.close()
        
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None