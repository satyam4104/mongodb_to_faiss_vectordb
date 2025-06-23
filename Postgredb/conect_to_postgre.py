import psycopg2
from psycopg2 import Error

try:
    # Establish connection to PostgreSQL
    connection = psycopg2.connect(
        host="localhost",        # Database host
        port="5432",            # Default PostgreSQL port
        database="postgres",        # Replace with your database name
        user="postgres",        # Replace with your username
        password="1331" # Replace with your password
    )

    # Create a cursor object to execute queries
    cursor = connection.cursor()

    # Example: Create a table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50),
            email VARCHAR(100)
        );
    """)

    # Example: Insert data
    cursor.execute("INSERT INTO users (name, email) VALUES (%s, %s)", ("John Doe", "john@example.com"))

    # Example: Query data
    cursor.execute("SELECT * FROM users;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    # Commit the transaction
    connection.commit()
    print("Operations completed successfully!")

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL:", error)

finally:
    # Close cursor and connection
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection closed.")