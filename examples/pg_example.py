from sqlalchemy import and_
from swarms_memory import PostgresDB

# Define the connection string for the PostgreSQL database
connection_string = "postgresql://user:password@localhost/mydatabase"

# Define the table name where vectors will be stored
table_name = "vectors_table"

# Create an instance of the PostgresDB class
db = PostgresDB(connection_string, table_name)

# Add a new vector to the database
vector_data = "1.0, 2.0, 3.0"  # Example vector as a string
db.add(
    vector=vector_data,
    namespace="example_namespace",
    meta={"source": "test"},
)

# Create a query condition (e.g., filter by a specific namespace)
query_condition = and_(
    db.VectorModel.vector.like("1.0%")
)  # Example condition
results = db.query(
    query=query_condition, namespace="example_namespace"
)

# Print the results of the query
for result in results:
    print(
        f"ID: {result.id}, Vector: {result.vector}, Namespace: {result.namespace}, Meta: {result.meta}"
    )

# Delete a vector from the database by its ID
# Assuming we have the ID of the vector we want to delete
vector_id_to_delete = (
    results[0].id if results else None
)  # Get the first result's ID
if vector_id_to_delete:
    db.delete_vector(vector_id_to_delete)
    print(f"Deleted vector with ID: {vector_id_to_delete}")
else:
    print("No vector found to delete.")
