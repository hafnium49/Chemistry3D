from openai import OpenAI, OpenAIError
import os
from datetime import datetime
from omni.isaac.examples.user_examples.Chemistry3D_utils import *
# from utils import *
import traceback
from dotenv import load_dotenv  # Import the load_dotenv function
import json
import ast

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__)) #os.getcwd()
# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Load environment variables from the .env file
load_dotenv(f"{parent_directory}/.env")

# Get the OpenAI API key from the .env file
api_key_value = os.getenv("OPENAI_API_KEY")
# print(f"API Key: {api_key_value}")
# Configure the global client
# openai.api_type = "your_own_type"
# openai.base_url = "your_own_base"  # Note: 'api_base' is now 'base_url'
# openai.api_version = "your_own_version"

class AgentLLM:
    def __init__(self, name: str, model_engine="gpt-4o", save_path=""):
        self._name = name
        self._model_engine = model_engine

        self.system_prompt = "You are a helpful AI assistant."
        self.conversation_log = []
        if save_path:
            self.save_path = save_path
        else:
            # Get the current directory
            current_directory = os.path.dirname(os.path.abspath(__file__))
            self.save_path = f'{current_directory}/log'
        # # Load environment variables from the .env file
        # load_dotenv("../.env")
        # api_key = os.getenv("OPENAI_API_KEY")
        # print(f"API Key: {api_key_value}")
        self.client = OpenAI(api_key=api_key_value)

    def load_system_prompt_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            self.system_prompt = f.read()

    def append_system_prompt_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            new_str = f.read()
        self.system_prompt += new_str

    def get_name(self):
        return self._name

    def generate_response(self, prompt, input_messages=None, retry_limit=3, max_tokens=10000, temperature=0.7, response_format=None):
        attempts = 0
        while attempts < retry_limit:
            try:
                if input_messages is None:
                    input_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    prompt = input_messages[-1]["content"]
                
                if response_format:
                    response = self.client.beta.chat.completions.parse(
                        model=self._model_engine,
                        messages=input_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=response_format
                    )
                    message = response.choices[0].message.parsed
                    self._append_to_log(prompt, str(message))
                    self._save_conversation()
                    print(f'{self._name}: Response has been generated successfully.')
                else:
                    response = self.client.chat.completions.create(
                        model=self._model_engine,
                        messages=input_messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    message = response.choices[0].message.content
                    self._append_to_log(prompt, message)
                    self._save_conversation()
                    print(f'{self._name}: Response has been generated successfully.')
                    return message.strip()
            except OpenAIError as e:
                attempts += 1
                print(f"Attempt {attempts}: An error occurred - {e}")

        print(f"All {retry_limit} retries failed.")
        return None

    def _append_to_log(self, prompt, response):
        self.conversation_log.append({'prompt': prompt, 'response': response})

    def _save_conversation(self):
        filepath = os.path.join(self.save_path, f"{self._name}_log.txt")

        # Clear the file
        with open(filepath, 'w') as f:
            pass

        with open(filepath, 'w') as f:
            for exchange in self.conversation_log:
                f.write(f"Prompt: {exchange['prompt']}\n")
                f.write(f"####################\n")
                f.write(f"Response: {exchange['response']}\n")
                f.write("\n")  # Add a newline for better readability
                f.write(f"####################\n\n")

    # def exec_code(self, code: str, global_dict: dict):
    #     # Preprocess code by replacing \\n with \n
    #     code = code.replace('  \\n  ', '\n').replace(' \\n ', '\n').replace('\\n', '\n')

    #     try:
    #         print(code)
    #         exec(code, global_dict)
    #         return True, ''
    #     except Exception as e:
    #         error_traceback = traceback.format_exc()
    #         print(f"An error occurred: {e}")
    #         return False, error_traceback

    # def exec_code(self, code: str, global_dict: dict):
    #     # Preprocess code by replacing \\n with \n and removing code block markers
    #     code = code.replace('  \\n  ', '\n').replace(' \\n ', '\n').replace('\\n', '\n').strip()
    #     if code.startswith("```"):
    #         code = code[3:].lstrip()
    #         if code.startswith(('json', 'python')):
    #             code = code.split('\n', 1)[1]
    #     if code.endswith("```"):
    #         code = code[:-3]

    #     try:
    #         # Parse the JSON content
    #         code_json = json.loads(code)
    #         # Extract the code to execute
    #         exec_code = code_json.get('Code', '')
    #         print(exec_code)
    #         exec(exec_code, global_dict)
    #         return True, ''
    #     except Exception as e:
    #         error_traceback = traceback.format_exc()
    #         print(f"An error occurred: {e}")
    #         return False, error_traceback

    def exec_code(self, code: str | dict, global_dict: dict):
        if isinstance(code, str):
            # Preprocess code by replacing \\n with \n and removing code block markers
            code = code.replace('  \\n  ', '\n').replace(' \\n ', '\n').replace('\\n', '\n').strip()
            if code.startswith("```"):
                code = code[3:].lstrip()
                if code.startswith(('json', 'python')):
                    code = code.split('\n', 1)[1]
            if code.endswith("```"):
                code = code[:-3]
            # Use ast.literal_eval to safely parse the code string
            code_dict = ast.literal_eval(code)
        elif isinstance(code, dict):
            code_dict = code
        else:
            print("Invalid code type")
            return False, "Invalid code type"

        try:
            # Extract the code to execute
            exec_code = code_dict.get('Code', '')
            print(exec_code)
            exec(exec_code, global_dict)
            return True, ''
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"An error occurred: {e}")
            return False, error_traceback

if __name__ == "__main__":
    agent = AgentLLM("test_agent")
    print(agent.generate_response("Hello"))
