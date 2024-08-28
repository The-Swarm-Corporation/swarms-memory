# Example usage of the DictInternalMemory class

# Import necessary libraries
import random
from swarms_memory import DictInternalMemory


# Create an instance of the DictInternalMemory class with a limit of 5 entries
memory = DictInternalMemory(n_entries=5)


# Function to simulate adding entries to the internal memory
def add_entries_to_memory(num_entries):
    for _ in range(num_entries):
        score = random.uniform(
            0, 100
        )  # Generate a random score between 0 and 100
        content = (
            f"Entry with score {score}"  # Create a content string
        )
        memory.add(
            score=score, entry=content
        )  # Add the entry to memory


# Add 10 entries to the internal memory
add_entries_to_memory(10)

# Retrieve and print the top 5 entries from the internal memory
top_entries = memory.get_top_n(n=5)
print("Top 5 entries in internal memory:")
for key, entry in top_entries:
    print(
        f"Key: {key}, Score: {entry['score']}, Content: {entry['content']}"
    )

# Print the total number of entries currently in memory
print(f"\nTotal entries in memory: {memory.len()}")

# Add more entries to see how the memory limits the number of stored entries
add_entries_to_memory(10)

# Retrieve and print the updated top 5 entries from the internal memory
updated_top_entries = memory.get_top_n(n=5)
print(
    "\nUpdated Top 5 entries in internal memory after adding more entries:"
)
for key, entry in updated_top_entries:
    print(
        f"Key: {key}, Score: {entry['score']}, Content: {entry['content']}"
    )

# Print the total number of entries currently in memory after updates
print(f"\nTotal entries in memory after updates: {memory.len()}")
