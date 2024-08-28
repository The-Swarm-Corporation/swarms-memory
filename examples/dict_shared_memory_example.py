# Example usage of the DictSharedMemory class

# Import necessary libraries
import time
from swarms_memory import DictSharedMemory
import threading

# Define the file location where the shared memory will be stored
file_location = "shared_memory.json"

# Create an instance of the DictSharedMemory class
shared_memory = DictSharedMemory(file_loc=file_location)


# Function to simulate adding entries to shared memory
def add_entries(agent_id, cycles):
    for cycle in range(cycles):
        score = cycle * 10.0  # Example score based on cycle
        entry_content = (
            f"Entry from agent {agent_id} at cycle {cycle}"
        )
        shared_memory.add(
            score=score,
            agent_id=agent_id,
            agent_cycle=cycle,
            entry=entry_content,
        )
        time.sleep(1)  # Simulate some delay


# Start multiple threads to add entries concurrently
thread1 = threading.Thread(target=add_entries, args=("Agent_1", 5))
thread2 = threading.Thread(target=add_entries, args=("Agent_2", 5))

thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()

# Retrieve the top 3 entries from the shared memory
top_entries = shared_memory.get_top_n(n=3)

# Print the top entries
print("Top 3 entries in shared memory:")
for entry_id, entry_data in top_entries.items():
    print(
        f"ID: {entry_id}, Agent: {entry_data['agent']}, Score: {entry_data['score']}, Cycle: {entry_data['cycle']}, Content: {entry_data['content']}"
    )

# At this point, the shared memory file 'shared_memory.json' will contain all the entries added
