# agent.py

from openai import OpenAI, OpenAIError
import os
from datetime import datetime
from omni.isaac.examples.user_examples.Chemistry3D_utils import *
# from omni.isaac.examples.user_examples.LLM.mas import GenerateControllers, AddControllers, AddTasks
from omni.isaac.examples.user_examples.tools import get_function_schemas, add_pickmove_task, add_pour_task, add_return_task
import traceback
from dotenv import load_dotenv
import json
import ast
from pydantic import BaseModel  # Import BaseModel for type checking
from typing import List, Dict, Any

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Load environment variables from the .env file
load_dotenv(f"{parent_directory}/.env")

# Get the OpenAI API key from the .env file
api_key_value = os.getenv("OPENAI_API_KEY")

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

    # def generate_response(self, prompt, input_messages=None, retry_limit=3, max_tokens=10000, temperature=0.7, response_format=None):
    #     attempts = 0
    #     while attempts < retry_limit:
    #         try:
    #             if input_messages is None:
    #                 input_messages = [
    #                     {"role": "system", "content": self.system_prompt},
    #                     {"role": "user", "content": prompt}
    #                 ]
    #                 prompt = input_messages[-1]["content"]
                
    #             if response_format:
    #                 response = self.client.beta.chat.completions.parse(
    #                     model=self._model_engine,
    #                     messages=input_messages,
    #                     max_tokens=max_tokens,
    #                     temperature=temperature,
    #                     response_format=response_format
    #                 )
    #                 message = response.choices[0].message.parsed
    #                 self._append_to_log(prompt, str(message))
    #                 self._save_conversation()
    #                 print(f'{self._name}: Response has been generated successfully.')
    #                 print(f"type: {type(message)}")
    #                 return message
    #             else:
    #                 response = self.client.chat.completions.create(
    #                     model=self._model_engine,
    #                     messages=input_messages,
    #                     max_tokens=max_tokens,
    #                     temperature=temperature
    #                 )
    #                 message = response.choices[0].message.content
    #                 self._append_to_log(prompt, message)
    #                 self._save_conversation()
    #                 print(f'{self._name}: Response has been generated successfully.')
    #                 return message.strip()
    #         except OpenAIError as e:
    #             attempts += 1
    #             print(f"Attempt {attempts}: An error occurred - {e}")

    #     print(f"All {retry_limit} retries failed.")
    #     return None

    def generate_response(self, prompt, input_messages=None, retry_limit=3, max_tokens=10000, temperature=0.7):
        attempts = 0
        while attempts < retry_limit:
            try:
                if input_messages is None:
                    input_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    prompt = input_messages[-1]["content"]
                
                function_schemas = get_function_schemas()

                response = self.client.chat.completions.create(
                    model=self._model_engine,
                    messages=input_messages,
                    tools=function_schemas,
                    # function_call="auto",
                    # max_tokens=max_tokens,
                    # temperature=temperature
                )

                message = response.choices[0].message

                # Check if the assistant wants to call a function
                if message.tool_calls:
                    self._append_to_log(prompt, str(message))
                    self._save_conversation()
                    print(f'{self._name}: Function call received.')
                    print(f'Content: {message.content}')
                    return message  # Return the message containing the function call
                else:
                    # Regular response
                    self._append_to_log(prompt, message.content)
                    self._save_conversation()
                    print(f'{self._name}: Response has been generated successfully.')
                    return message.content #message.get("content", "").strip()
            except OpenAIError as e:
                attempts += 1
                print(f"Attempt {attempts}: An error occurred - {e}")

        print(f"All {retry_limit} retries failed.")
        return None

    def handle_function_call(self, tool_call, global_dict, controller_manager, current_observations, robot):
        function_name = tool_call.function.name
        arguments_str = tool_call.function.arguments
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            print(f"Error decoding arguments: {e}")
            return None

        print(f"Function name: {function_name}")
        print(f"Arguments: {arguments}")

        # Map function names to actual functions
        function_mapping = {
            "add_pickmove_task": add_pickmove_task,
            "add_pour_task": add_pour_task,
            "add_return_task": add_return_task
        }

        if function_name in function_mapping:
            function_to_call = function_mapping[function_name]

            if function_name == "add_pickmove_task":
                print("Calling add_pickmove_task")
                # Extract arguments
                picking_object = arguments.get("picking_object")
                target_object = arguments.get("target_object")
                target_position = arguments.get("target_position")
                # end_effector_offset = arguments.get("end_effector_offset")
                # end_effector_orientation = arguments.get("end_effector_orientation")
                # current_joint_positions = arguments.get("current_joint_positions")
                # If current_joint_positions is None, get from robot
                if current_joint_positions is None:
                    current_joint_positions = robot.get_joint_positions().tolist()
                # Call the function with all arguments
                result = function_to_call(
                    controller_manager=controller_manager,
                    picking_object=picking_object,
                    target_object=target_object,
                    target_position=target_position,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=end_effector_offset,
                    end_effector_orientation=end_effector_orientation,
                    current_observations=current_observations,
                    robot=robot
                )
                return result

            elif function_name == "add_pour_task":
                print("Calling add_pour_task")
                # Extract arguments
                pour_speed = arguments.get("pour_speed")
                current_joint_positions = arguments.get("current_joint_positions")
                current_joint_velocities = arguments.get("current_joint_velocities")
                if current_joint_positions is None:
                    current_joint_positions = robot.get_joint_positions().tolist()
                if current_joint_velocities is None:
                    current_joint_velocities = robot.get_joint_velocities().tolist()
                # Call the function
                result = function_to_call(
                    controller_manager=controller_manager,
                    pour_speed=pour_speed,
                    current_joint_positions=current_joint_positions,
                    current_joint_velocities=current_joint_velocities,
                    current_observations=current_observations,
                    robot=robot
                )
                return result

            elif function_name == "add_return_task":
                print("Calling add_return_task")
                # Extract arguments
                picking_object = arguments.get("picking_object")
                # end_effector_offset = arguments.get("end_effector_offset")
                # end_effector_orientation = arguments.get("end_effector_orientation")
                # current_joint_positions = arguments.get("current_joint_positions")
                if current_joint_positions is None:
                    current_joint_positions = robot.get_joint_positions().tolist()
                # Call the function
                result = function_to_call(
                    controller_manager=controller_manager,
                    picking_object=picking_object,
                    current_joint_positions=current_joint_positions,
                    end_effector_offset=end_effector_offset,
                    end_effector_orientation=end_effector_orientation,
                    current_observations=current_observations,
                    robot=robot
                )
                return result

        else:
            print(f"Function {function_name} not found.")
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

    def exec_code(self, code, global_dict: dict):
        """
        Execute the code provided, handling different types of inputs:
        - If code is a string, attempt to parse it and extract the 'Code' key.
        - If code is a dict, extract the 'Code' key.
        - If code is a Pydantic BaseModel instance (e.g., GenerateControllers or AddControllers), extract the 'code' attribute.

        Args:
            code: The code to execute, which can be a string, dict, or BaseModel instance.
            global_dict: The global dictionary in which to execute the code.

        Returns:
            A tuple (success: bool, error_message: str)
        """
        if isinstance(code, str):
            # Preprocess code by replacing \\n with \n and removing code block markers
            code_clean = code.replace('  \\n  ', '\n').replace(' \\n ', '\n').replace('\\n', '\n').strip()
            if code_clean.startswith("```"):
                code_clean = code_clean[3:].lstrip()
                if code_clean.startswith(('json', 'python')):
                    code_clean = code_clean.split('\n', 1)[1]
            if code_clean.endswith("```"):
                code_clean = code_clean[:-3]

            # Try to parse the code string as JSON
            try:
                code_dict = json.loads(code_clean)
                exec_code = code_dict.get('Code', '')
            except json.JSONDecodeError:
                # If JSON parsing fails, try to parse as Python literal
                try:
                    code_dict = ast.literal_eval(code_clean)
                    exec_code = code_dict.get('Code', '')
                except Exception:
                    # If parsing fails, assume code is directly executable
                    exec_code = code_clean
        # elif isinstance(code, dict):
        #     exec_code = code.get('Code', '')
        elif isinstance(code, BaseModel):
            try:
                exec_code = code.step
            except AttributeError:
                try:
                    # Handle Pydantic models (GenerateControllers or AddControllers)
                    exec_code = code.code
                except AttributeError:
                    print("Invalid Pydantic model")
                    return False, "Invalid Pydantic model"
        else:
            print("Invalid code type")
            return False, "Invalid code type"

        try:
            # if exec_code has string type
            if isinstance(exec_code, str):
                print(exec_code)
                exec(exec_code, global_dict)
                return True, ''
            # if exec_code has list type
            elif isinstance(exec_code, list):
                for k, step in enumerate(exec_code):
                    print(f"Step {k+1}: {step.step_description}\n{step.code}")
                    exec(step.code, global_dict)
                return True, ''
            else:
                return False, f"Invalid code type: {type(exec_code)}"
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"An error occurred: {e}")
            return False, error_traceback

if __name__ == "__main__":
    agent = AgentLLM("test_agent")
    print(agent.generate_response("Hello"))
