# Example usage of the AbstractDatabase class

# First, we need to create a concrete implementation of the AbstractDatabase class
# Let's create a simple in-memory SQLite database as an example.

import sqlite3
from swarms_memory import AbstractDatabase


class SQLiteDatabase(AbstractDatabase):
    """
    Concrete implementation of the AbstractDatabase for SQLite.
    """

    def __init__(self):
        self.connection = None

    def connect(self):
        """Establishes a connection to the SQLite database."""
        self.connection = sqlite3.connect(
            ":memory:"
        )  # Using an in-memory database for demonstration
        self.create_table()

    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()

    def create_table(self):
        """Creates a sample table for demonstration purposes."""
        with self.connection:
            self.connection.execute(
                """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL
            )
            """
            )

    def execute_query(self, query):
        """Executes a database query."""
        with self.connection:
            return self.connection.execute(query)

    def fetch_all(self):
        """Fetches all rows from the result set."""
        return self.connection.fetchall()

    def fetch_one(self):
        """Fetches one row from the result set."""
        return self.connection.fetchone()

    def add(self, table, data):
        """Adds a new record to the database."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        query = (
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        )
        with self.connection:
            self.connection.execute(query, tuple(data.values()))

    def query(self, table, condition):
        """Queries the database based on the given condition."""
        query = f"SELECT * FROM {table} WHERE {condition}"
        self.execute_query(query)
        return self.fetch_all()

    def get(self, table, id):
        """Retrieves a record from the database based on the given ID."""
        query = f"SELECT * FROM {table} WHERE id = ?"
        self.execute_query(query, (id,))
        return self.fetch_one()

    def update(self, table, id, data):
        """Updates a record in the database based on the given ID."""
        set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE id = ?"
        with self.connection:
            self.connection.execute(query, (*data.values(), id))

    def delete(self, table, id):
        """Deletes a record from the database based on the given ID."""
        query = f"DELETE FROM {table} WHERE id = ?"
        with self.connection:
            self.connection.execute(query, (id,))


# Example usage of the SQLiteDatabase class
db = SQLiteDatabase()
db.connect()

# Adding new users to the database
db.add("users", {"name": "Alice", "age": 30})
db.add("users", {"name": "Bob", "age": 25})

# Querying users from the database
users = db.query("users", "age > 20")
print("Users older than 20:")
for user in users:
    print(f"ID: {user[0]}, Name: {user[1]}, Age: {user[2]}")

# Updating a user
db.update("users", 1, {"age": 31})  # Update Alice's age to 31

# Fetching a specific user
user = db.get("users", 1)
print(
    f"\nUpdated User: ID: {user[0]}, Name: {user[1]}, Age: {user[2]}"
)

# Deleting a user
db.delete("users", 2)  # Delete Bob

# Querying again to see remaining users
remaining_users = db.query("users", "1=1")  # Get all users
print("\nRemaining Users:")
for user in remaining_users:
    print(f"ID: {user[0]}, Name: {user[1]}, Age: {user[2]}")

# Closing the database connection
db.close()
