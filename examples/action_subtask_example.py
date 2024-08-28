# Example usage of the ActionSubtaskEntry class

# Import the ActionSubtaskEntry model
from swarms_memory import ActionSubtaskEntry


# Create an instance of the ActionSubtaskEntry class
def main():
    # Create a new action subtask entry with thought, action, and answer attributes
    action_entry = ActionSubtaskEntry(
        thought="I need to calculate the sum of two numbers.",
        action='{"tool": "calculator", "operation": "add", "numbers": [5, 3]}',
        answer="The sum of 5 and 3 is 8.",
    )

    # Print the attributes of the action entry
    print("Thought:", action_entry.thought)
    print("Action:", action_entry.action)
    print("Answer:", action_entry.answer)

    # Validate the instance and print its data as a dictionary
    entry_data = action_entry.dict()
    print("Entry Data as Dictionary:", entry_data)


# Entry point of the script
if __name__ == "__main__":
    main()
