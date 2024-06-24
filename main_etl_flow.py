import mysql.connector
import pandas as pd
import numpy as np
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import pickle
from sqlalchemy import create_engine
import csv

# Function to create MySQL connection
def create_connection(host, database, user, password):
    try:
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        if connection.is_connected():
            print(f"Connected to MySQL database: {database}")
        return connection
    except mysql.connector.Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Function to create table
def create_table(cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS clothes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Brand VARCHAR(255),
        Category VARCHAR(255),
        Color VARCHAR(255),
        Size VARCHAR(50),
        Material VARCHAR(255),
        Price DECIMAL(10, 2),
        LGBM_Prediction DECIMAL(10, 2),
        CatBoost_Prediction DECIMAL(10, 2)
    );
    """
    cursor.execute(create_table_query)
    print("Table 'clothes' created successfully.")

# Task to load and preprocess the data
@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_data():
    data = pd.read_csv('clothes_price_prediction_data.csv')
    return data

# Task to encode categorical data
@task
def encode_data(data, columns):
    for col in columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    return data

# Task to load models
@task
def load_models():
    with open('lightgbm_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    with open('catboost_model.pkl', 'rb') as f:
        cat_model = pickle.load(f)
    return lgb_model, cat_model

# Task for feature engineering
@task
def feature_engineering(data, models):
    lgb_model, cat_model = models
    X = data.drop('Price', axis=1)
    y = np.log1p(data['Price'])

    # Create predictions
    data['LGBM_Prediction'] = np.expm1(lgb_model.predict(X))
    data['CatBoost_Prediction'] = np.expm1(cat_model.predict(X))
    
    # Add more feature engineering logic here
    return data

# Task to load data into MySQL
@task
def load_to_mysql(data):
    host = 'localhost'
    database = 'sales_product'
    user = 'root'
    password = ''
    
    connection = create_connection(host, database, user, password)
    if connection:
        cursor = connection.cursor()
        create_table(cursor)
        
        # Insert data into the table
        for _, row in data.iterrows():
            cursor.execute("""
            INSERT INTO clothes (Brand, Category, Color, Size, Material, Price, LGBM_Prediction, CatBoost_Prediction) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
        
        connection.commit()
        cursor.close()
        connection.close()
        print("Data inserted successfully into MySQL.")

# Define the flow
@flow
def etl_flow():
    data = load_data()
    encoded_data = encode_data(data, ['Brand', 'Category', 'Color', 'Size', 'Material'])
    models = load_models()
    transformed_data = feature_engineering(encoded_data, models)
    load_to_mysql(transformed_data)

# Run the flow
if __name__ == "__main__":
    etl_flow()
