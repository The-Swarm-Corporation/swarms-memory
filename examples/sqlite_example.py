# Example usage of the SQLiteDB class
from swarms_memory import SQLiteDB

# Define the path to the SQLite database file
db_path = "example_database.db"

# Create an instance of the SQLiteDB class
db = SQLiteDB(db_path)

# Create a table for storing example data (if it doesn't already exist)
create_table_query = """
CREATE TABLE IF NOT EXISTS example_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL
);
"""
db.execute_query(create_table_query)

# Add a new entry to the database
insert_query = "INSERT INTO example_table (name, age) VALUES (?, ?)"
db.add(insert_query, ("Alice", 30))

# Add another entry to the database
db.add(insert_query, ("Bob", 25))

# Query the database to retrieve all entries
select_query = "SELECT * FROM example_table"
results = db.query(select_query)

# Print the results of the query
print("Entries in the database:")
for result in results:
    print(f"ID: {result[0]}, Name: {result[1]}, Age: {result[2]}")

# Update an entry in the database
update_query = "UPDATE example_table SET age = ? WHERE name = ?"
db.update(update_query, (31, "Alice"))

# Query the database again to see the updated entry
results = db.query(select_query)
print("\nEntries after update:")
for result in results:
    print(f"ID: {result[0]}, Name: {result[1]}, Age: {result[2]}")

# Delete an entry from the database
delete_query = "DELETE FROM example_table WHERE name = ?"
db.delete(delete_query, ("Bob",))

# Query the database to see the remaining entries
results = db.query(select_query)
print("\nEntries after deletion:")
for result in results:
    print(f"ID: {result[0]}, Name: {result[1]}, Age: {result[2]}")
