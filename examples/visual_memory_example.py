# Example usage of the VisualShortTermMemory class

# Import necessary libraries
from datetime import datetime
from swarms_memory import VisualShortTermMemory

# Create an instance of the VisualShortTermMemory class
visual_memory = VisualShortTermMemory()

# Define some example images, descriptions, timestamps, and locations
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
descriptions = ["A beautiful sunset", "A busy street", "A calm beach"]
timestamps = [
    datetime.now().timestamp(),
    datetime.now().timestamp() + 1,
    datetime.now().timestamp() + 2,
]
locations = ["Beach", "City", "Beach"]

# Add the images and their details to the visual memory
visual_memory.add(
    images=images,
    description=descriptions,
    timestamps=timestamps,
    locations=locations,
)

# Retrieve and print all images stored in memory
stored_images = visual_memory.get_images()
print("Stored Images:", stored_images)

# Retrieve and print all descriptions stored in memory
stored_descriptions = visual_memory.get_descriptions()
print("Stored Descriptions:", stored_descriptions)

# Search for images captured at a specific location
location_search = "Beach"
images_at_location = visual_memory.search_by_location(
    location=location_search
)
print(f"Images captured at {location_search}:", images_at_location)

# Search for images captured within a specific time range
# Assuming we want to search for images added in the last 2 seconds
start_time = datetime.now().timestamp() - 3  # 3 seconds ago
end_time = datetime.now().timestamp() + 3  # 3 seconds from now
images_in_time_range = visual_memory.search_by_timestamp(
    start_time=start_time, end_time=end_time
)
print("Images captured in the last 2 seconds:", images_in_time_range)

# Print the entire memory as a string representation
memory_string = visual_memory.return_as_string()
print("Memory as string:", memory_string)
