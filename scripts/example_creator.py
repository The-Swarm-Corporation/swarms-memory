import os
from swarms import Agent, OpenAIChat, extract_code_from_markdown
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class DocumentationGenerator:
    """
    A class that generates documentation for Python files in a specified folder.

    Attributes:
        agent (Agent): The agent responsible for running the documentation generation process.
        folder_name (str): The name of the folder containing the Python files.
        examples_folder (str): The name of the folder where usage examples will be saved.

    Methods:
        get_python_files(): Retrieve all Python files in the specified folder, excluding __init__.py.
        extract_code_from_file(file_path): Extract all the code from a Python file.
        generate_documentation(code): Generate a usage example for the given code.
        save_example(file_name, usage_example): Save the generated usage example to a file.
        run(): Generate documentation for all Python files in the folder.
    """

    def __init__(self, agent: Agent, folder_name: str):
        self.agent = agent
        self.folder_name = folder_name
        self.examples_folder = "examples"
        os.makedirs(self.examples_folder, exist_ok=True)

    def get_python_files(self):
        """Retrieve all Python files in the specified folder, excluding __init__.py."""
        python_files = []
        for root, _, files in os.walk(self.folder_name):
            for file in files:
                if file.endswith(".py") and file != "__init__.py":
                    python_files.append(os.path.join(root, file))
        return python_files

    def extract_code_from_file(self, file_path):
        """Extract all the code from a Python file."""
        with open(file_path, "r") as file:
            code = file.read()
        return code

    def generate_documentation(self, code):
        """Generate a usage example for the given code."""
        prompt = f"Generate a usage example for the following Python code:\n\n{code}\n\nProvide detailed comments."
        code = self.agent.run(prompt)
        return extract_code_from_markdown(code)

    def save_example(self, file_name, usage_example):
        """Save the generated usage example to a file in the examples folder."""
        example_file_path = os.path.join(
            self.examples_folder, file_name
        )
        with open(example_file_path, "w") as file:
            file.write(usage_example)

    def run(self):
        """Generate documentation for all Python files in the folder."""
        python_files = self.get_python_files()
        for file_path in python_files:
            logger.info(f"Processing file: {file_path}")
            code = self.extract_code_from_file(file_path)
            usage_example = self.generate_documentation(code)
            logger.info(f"Usage example generated for {file_path}")
            example_file_name = os.path.basename(file_path).replace(
                ".py", "_example.py"
            )
            self.save_example(example_file_name, usage_example)
            logger.success(
                f"Documentation saved for {file_path} as {example_file_name}"
            )
            logger.success(
                f"Documentation saved for {file_path} as {example_file_name}"
            )


# Get the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the OpenAIChat class
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

# Initialize the agent
agent = Agent(
    agent_name="ExampleCreator",
    system_prompt="Create a usage example for the following Python code: \n\n Return only python code.",
    llm=model,
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="finance_agent.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Initialize the DocumentationGenerator with the folder name
folder_name = "swarms_memory"
doc_generator = DocumentationGenerator(
    agent=agent, folder_name=folder_name
)

# Create documentation
doc_generator.run()
