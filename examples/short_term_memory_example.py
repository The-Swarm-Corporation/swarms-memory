# Example usage of the ShortTermMemory class

# Import necessary libraries
from swarms_memory import ShortTermMemory

# Create an instance of the ShortTermMemory class
stm = ShortTermMemory()

# Add messages to the short term memory
stm.add(role="agent", message="Hello world!")
stm.add(role="agent", message="How are you?")
stm.add(role="agent", message="I am fine.")
stm.add(role="agent", message="What about you?")
stm.add(role="agent", message="I am doing well, thank you!")

# Retrieve and print the short term memory
short_term_memory = stm.get_short_term()
print("Short Term Memory:")
for entry in short_term_memory:
    print(f"Role: {entry['role']}, Message: {entry['message']}")

# Search for a specific term in the memory
search_term = "fine"
search_results = stm.search_memory(search_term)
print(f"\nSearch results for term '{search_term}':")
print("Short Term Results:", search_results["short_term"])
print("Medium Term Results:", search_results["medium_term"])

# Move the first message from short term to medium term memory
if short_term_memory:
    stm.move_to_medium_term(0)

# Retrieve and print the medium term memory
medium_term_memory = stm.get_medium_term()
print("\nMedium Term Memory:")
for entry in medium_term_memory:
    print(f"Role: {entry['role']}, Message: {entry['message']}")

# Save the current memory to a file
filename = "memory.json"
stm.save_to_file(filename)

# Clear the short term memory
stm.clear()

# Load the memory back from the file
stm.load_from_file(filename)

# Print the loaded short term memory
loaded_short_term_memory = stm.get_short_term()
print("\nLoaded Short Term Memory from file:")
for entry in loaded_short_term_memory:
    print(f"Role: {entry['role']}, Message: {entry['message']}")

# Clear the medium term memory
stm.clear_medium_term()

# Print the medium term memory after clearing
print("\nMedium Term Memory after clearing:")
print(stm.return_medium_memory_as_str())
