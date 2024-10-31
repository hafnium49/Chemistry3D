# mas.py

from omni.isaac.examples.user_examples.LLM.agent import AgentLLM as Agent
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
import functools
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Controllers.pick_move_controller import PickMoveController
from omni.isaac.examples.user_examples.Controllers.pour_controller import PourController
from omni.isaac.examples.user_examples.Controllers.return_controller import PlaceController as ReturnController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.examples.user_examples.chem_sim.simulation.database import reactions

import os
import numpy as np
from pxr import Sdf, Gf, UsdPhysics
from omni.isaac.examples.user_examples.Sim_Container import Sim_Container
import traceback
from pydantic import BaseModel, Field

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)

PROMPTS_PATH = f'{parent_directory}/prompts_JSON'
LOG_PATH = f'{current_directory}/log'

class GenerateControllers(BaseModel):
    """
    ```json
    {
        "Controllers' Name": "PickMoveController",
        "Task Description": "Controls to pick up the left beaker, which is the beaker_Kmno4, and move it to a specific position.",
        "Code": "pickmove_controller = PickMoveController(name='pickmove_controller', cspace_controller=RMPFlowController(name='pickmove_cspace_controller', robot_articulation=Franka), gripper=Franka.gripper, speed=1.5)"
    }
    ```
    """
    controllers_name: str = Field(alias="Controllers' Name")
    task_description: str = Field(alias="Task Description")
    code: str = Field(alias="Code")

    class Config:
        populate_by_name = True

class AddControllers(BaseModel):
    """
    ```json
    {
        "Task Description": "Control the robotic arm to pick up beaker_Kmno4 using the Franka robot.",
        "Code": "controller_manager.add_controller('pickmove_controller', pickmove_controller)"
    }
    ```
    """
    task_description: str = Field(alias="Task Description")
    code: str = Field(alias="Code")

    class Config:
        populate_by_name = True

class Step(BaseModel):
    step_description: str = Field(alias="Step Description")
    code: str = Field(alias="Code")

    class Config:
        populate_by_name = True

class AddTasks(BaseModel):
    """
    ```json
    {
        "Task Description": "Control the robotic arm to pick up beaker_Kmno4 using the Franka robot. Then pour the contents and return the beaker to its original position.",
        "Step": ["controller_manager.add_task('pickmove_controller', {'picking_position': lambda obs: obs['beaker_Kmno4']['position'], 'target_position': lambda obs: obs['beaker_Kmno4']['Pour_Position'], 'current_joint_positions': lambda obs: Franka.get_joint_positions()})", "controller_manager.add_task('pour_controller', {'franka_art_controller': lambda obs: Franka.get_articulation_controller(), 'current_joint_positions': lambda obs: Franka.get_joint_positions(), 'current_joint_velocities': lambda obs: Franka.get_joint_velocities(), 'pour_speed': 55 / 180.0 * np.pi})", "controller_manager.add_task('return_controller', {'pour_position': lambda obs: obs['beaker_Kmno4']['Pour_Position'], 'return_position': lambda obs: np.array(obs['beaker_Kmno4']['Return_Position']), 'current_joint_positions': lambda obs: Franka.get_joint_positions()})"
    }
    ```
    """

    task_description: str = Field(alias="Task Description")
    step: list[Step] = Field(alias="Step")

    class Config:
        populate_by_name = True

class MAS:
    """
    Multi-Agent System (MAS) for managing and simulating chemical reactions and robotics control in a simulated environment.

    Args:
        world: The simulation world object.
        controller_manager: The controller manager object.
    """

    def __init__(self, world, controller_manager) -> None:
        self.my_world = world
        self._observation = self.my_world.get_observations()

        self.plan_steps_list = []
        self.plan_message_str = ''  # Used for debugging
        self.code_str = ''
        self.reaction_dict = reactions
        self.generated_func_str = ''

        self.max_num_retry = 3
        self.utils = Utils()
        self.utils._set_particle_parameter(self.my_world)
        self.initial_coder_function_dict = {
            'scenePath': Sdf.Path("/physicsScene"),
            'Gf': Gf,
            'UsdPhysics': UsdPhysics,
            'my_world': self.my_world,
            'current_observations': self.my_world.get_observations(),
            'Franka': self.my_world.scene.get_object("Franka"),
            'Bottle_Kmno4': self.my_world.scene.get_object("Bottle_Kmno4"),
            'beaker_Kmno4': self.my_world.scene.get_object("beaker_Kmno4"),
            'Bottle_Fecl2': self.my_world.scene.get_object("Bottle_Fecl2"),
            'beaker_Fecl2': self.my_world.scene.get_object("beaker_Fecl2"),
            'Feo': self.my_world.scene.get_object("Feo"),
            'Sim_Container': Sim_Container,
            'controller_manager': controller_manager,
            'PickMoveController': PickMoveController,
            'PourController': PourController,
            'ReturnController': ReturnController,
            'RMPFlowController': RMPFlowController,
            'euler_angles_to_quat': euler_angles_to_quat,
            'np': np,
            'utils': self.utils,
        }
        self.coder_function_dict = self.initial_coder_function_dict.copy()

        self.planner_prompt_filename = '/planner_prompt.txt'
        self.agents_initialization()

    def agents_initialization(self):
        """
        Initialize agents for different tasks and load their system prompts.
        """
        self.agent_controller_generator = Agent("controller_generator", save_path=LOG_PATH)
        self.agent_planner = Agent("planner", save_path=LOG_PATH)
        self.agent_coder = Agent("coder", save_path=LOG_PATH)
        self.agent_debugger = Agent("debugger", save_path=LOG_PATH)
        self.agent_reaction_responser = Agent("reaction_responser", save_path=LOG_PATH)
        self.agent_add_controllers = Agent("add_controllers", save_path=LOG_PATH)
        self.agent_add_sim_containers = Agent("add_sim_containers", save_path=LOG_PATH)
        self.agent_add_rigidbody = Agent("add_rigidbody", save_path=LOG_PATH)
        self.agent_add_particles = Agent("add_particles", save_path=LOG_PATH)
        self.agent_add_tasks = Agent("add_tasks", save_path=LOG_PATH)
        self.agent_assistant = Agent("assistant", save_path=LOG_PATH)

        # Load system prompts for other agents
        self.agent_controller_generator.load_system_prompt_from_file(PROMPTS_PATH + '/controller_generator_prompt.txt')
        self.agent_reaction_responser.load_system_prompt_from_file(PROMPTS_PATH + '/reaction_responser_prompt.txt')
        self.agent_add_controllers.load_system_prompt_from_file(PROMPTS_PATH + '/add_controller_prompt.txt')
        self.agent_add_sim_containers.load_system_prompt_from_file(PROMPTS_PATH + '/add_sim_container_prompt.txt')
        self.agent_add_rigidbody.load_system_prompt_from_file(PROMPTS_PATH + '/add_rigid_body_prompt.txt')
        self.agent_add_particles.load_system_prompt_from_file(PROMPTS_PATH + '/add_particle_set_prompt.txt')
        self.agent_add_tasks.load_system_prompt_from_file(PROMPTS_PATH + '/add_tasks_prompt.txt')
        self.agent_coder.load_system_prompt_from_file(PROMPTS_PATH + '/coder_prompt.txt')
        self.agent_debugger.load_system_prompt_from_file(PROMPTS_PATH + '/debugger_prompt.txt')
        self.agent_assistant.load_system_prompt_from_file(PROMPTS_PATH + '/assistant_prompt.txt')
        

    def _update_system_prompts(self):
        """
        Update system prompts for all agents.
        """
        self.agent_controller_generator.load_system_prompt_from_file(PROMPTS_PATH + '/controller_generator_prompt.txt')
        self.agent_reaction_responser.load_system_prompt_from_file(PROMPTS_PATH + '/reaction_responser_prompt.txt')
        self.agent_add_controllers.load_system_prompt_from_file(PROMPTS_PATH + '/add_controller_prompt.txt')
        self.agent_add_sim_containers.load_system_prompt_from_file(PROMPTS_PATH + '/add_sim_container_prompt.txt')
        self.agent_add_rigidbody.load_system_prompt_from_file(PROMPTS_PATH + '/add_rigid_body_prompt.txt')
        self.agent_add_particles.load_system_prompt_from_file(PROMPTS_PATH + '/add_particle_set_prompt.txt')
        self.agent_add_tasks.load_system_prompt_from_file(PROMPTS_PATH + '/add_tasks_prompt.txt')
        self.agent_coder.load_system_prompt_from_file(PROMPTS_PATH + '/coder_prompt.txt')
        self.agent_debugger.load_system_prompt_from_file(PROMPTS_PATH + '/debugger_prompt.txt')

    def _generate_plan(self, controllers_str):
        """
        Generate a plan for a given task.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            None
        """
        user_prompt = controllers_str
        message = self.agent_planner.generate_response(user_prompt)
        self.plan_steps_list = self.utils.extract_scripts(message)
        print(f'Number of generated plan steps: {len(self.plan_steps_list)}')

    def _debug_code(self, error_str, num_iter=3) -> bool:
        for i in range(num_iter):
            user_prompt = self.code_str + str(error_str)
            user_prompt += f"\n'observation: '{self.observation_str}"
            debug_code_str = self.agent_debugger.generate_response(user_prompt)
            flag, error_str = self.agent_coder.exec_code(debug_code_str, self.coder_function_dict)
            if flag:
                print("Debug successfully!")
                return True
        return False

    def _response_reaction(self, expected_chem):
        """
        Respond to a chemical reaction.

        Args:
            expected_chem (str): The expected chemical reaction.

        Returns:
            str: The response message.
        """
        observation = "observation: " + str(self._observation.keys()) + '\\n'
        user_prompt = observation + 'reaction_dict:' + str(self.reaction_dict) + '\\n' + expected_chem
        message = self.agent_reaction_responser.generate_response(user_prompt)
        return message

    def _add_particles(self):
        """
        Add particles for each object.

        Returns:
            str: The response message.
        """
        observation = str(self._observation)
        message = self.agent_add_particles.generate_response(observation)
        return message

    def _add_sim_container(self, particle_set_str):
        """
        Add simulation containers for each object.

        Args:
            particle_set_str (str): The string describing the particle set.

        Returns:
            str: The response message.
        """
        observation = "observation: " + str(self._observation) + '\\n'
        user_prompt = observation + particle_set_str
        message = self.agent_add_sim_containers.generate_response(user_prompt)
        return message

    def _add_rigidbody(self):
        """
        Add rigidbody for each object.

        Returns:
            str: The response message.
        """
        added_objects_dict = self.get_added_coder_function_dict()
        added_objects_dict_str = 'Objects introduced in the scene: ' + str(added_objects_dict.keys()) + '\\n'
        message = self.agent_add_rigidbody.generate_response(added_objects_dict_str)
        return message

    def _add_controllers(self, controllers_str):
        """
        Add controllers for a given task.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            AddControllers: The parsed response containing the code to execute.
        """
        self.observation_str = self._observations_to_string(self._observation)
        user_prompt = f"'observation: '{self.observation_str}\n{controllers_str}"

        # Generate response using the AddControllers model as the response format
        message = self.agent_add_controllers.generate_response(
            user_prompt,
            response_format=AddControllers
        )

        return message

    def _add_tasks(self, controllers_str):
        """
        Add tasks for a given controller.

        Args:
            controllers_str (str): The input string describing the controllers.

        Returns:
            AddControllers: The parsed response containing the code to execute.
        """
        self.observation_str = self._observations_to_string(self._observation)
        user_prompt = f"'observation: '{self.observation_str}\n{controllers_str}"

        # Generate response using the AddControllers model as the response format
        message = self.agent_add_tasks.generate_response(
            user_prompt,
            response_format=AddTasks  # Corrected the typo here
        )

        return message

    def _generate_controllers(self, prompt, observation):
        """
        Generate controllers for a given task.

        Args:
            prompt (str): The task description.
            observation (dict): The current observations.

        Returns:
            GenerateControllers: The generated controllers data.
        """
        self.observation_str = self._observations_to_string(observation)
        total_prompt = f"'observation: '{self.observation_str}\n{prompt}"

        # Generate response using the GenerateControllers model as the response format
        message = self.agent_controller_generator.generate_response(
            total_prompt,
            response_format=GenerateControllers
        )
        return message

    def _observations_to_string(self, observation):
        """
        Convert observation dictionary into a descriptive string.

        Args:
            observation (dict): The current observations.

        Returns:
            str: The observation
        """
        observation_str = ""
        for key, value in observation.items():
            observation_str += f"{key}: {value}\n"
        return observation_str

    def get_added_coder_function_dict(self):
        """
        Get the additional objects added to the coder function dictionary.

        Returns:
            dict: The added objects.
        """
        added_keys = set(self.coder_function_dict.keys()) - set(self.initial_coder_function_dict.keys())
        added_objects = {key: self.coder_function_dict[key] for key in added_keys}
        return added_objects

    def _generate_code_str(self, code_str):
        """
        Store the generated code string.

        Args:
            code_str (str): The code string to be stored.
        """
        if not hasattr(self, 'code_str'):
            self.code_str = ''
        self.code_str += code_str + '\n'

    def _execute_code_str(self, code=None):
        """
        Execute the stored code string.
        """
        if not code:
            code = self.code_str
        flag, error_str = self.agent_coder.exec_code(code, self.coder_function_dict)
        if not flag:
            # Handle error, perhaps debug
            print(f"Error executing code: {error_str}")
            # Optionally attempt to debug
            if not self._debug_code(error_str):
                print("Failed to debug code.")
        # Clear code_str after execution
        self.code_str = ''
