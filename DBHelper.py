import os
import pyodbc

class SQLDatabase:
    def __init__(self, server, database, username, password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = None

    def connect(self):
        try:
            conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}"
            self.connection = pyodbc.connect(conn_str)
        except pyodbc.Error as e:
            print("Error: Failed to connect to the database:", e)
            raise

    def execute_query(self, query, params=None):
        try:
            with self.connection.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                self.connection.commit()
        except pyodbc.Error as e:
            print("Error: Failed to execute query:", e)
            self.connection.rollback()
            raise

    def create_table(self, table_name, columns):
        # Assuming columns is a dictionary with column names and data types
        column_defs = ', '.join([f"{column} {data_type}" for column, data_type in columns.items()])
        query = f"CREATE TABLE {table_name} ({column_defs})"
        self.execute_query(query)

    def insert_record(self, table_name, record):
        column_names = ', '.join(record.keys())
        placeholders = ', '.join(['?'] * len(record))
        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
        self.execute_query(query, tuple(record.values()))

    def database_exists(self):
        query = f"SELECT name FROM sys.databases WHERE name = '{self.database}'"
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.rowcount > 0

    def table_exists(self, table_name):
        query = f"SELECT OBJECT_ID('{table_name}', 'U')"
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone() is not None

    def close_connection(self):
        if self.connection:
            self.connection.close()
            self.connection = None
