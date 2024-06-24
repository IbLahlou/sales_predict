import mysql.connector
import csv
from mysql.connector import Error

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
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def create_table(cursor):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS clothes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Brand VARCHAR(255),
        Category VARCHAR(255),
        Color VARCHAR(255),
        Size VARCHAR(50),
        Material VARCHAR(255),
        Price DECIMAL(10, 2)
    );
    """
    cursor.execute(create_table_query)
    print("Table 'clothes' created successfully.")

def insert_data_from_csv(connection, csv_file_path):
    cursor = connection.cursor()
    create_table(cursor)
    
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            cursor.execute("""
            INSERT INTO clothes (Brand, Category, Color, Size, Material, Price) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """, row)
        connection.commit()
        print("Data inserted successfully from CSV file.")

def main():
    # Remplacez les valeurs par vos informations de connexion MySQL
    host = 'localhost'
    database = 'sales_product'
    user = 'root'
    password = ''
    csv_file_path = 'clothes_price_prediction_data.csv'

    connection = create_connection(host, database, user, password)
    if connection:
        insert_data_from_csv(connection, csv_file_path)
        connection.close()
        print("MySQL connection is closed")

if __name__ == "__main__":
    main()
