# chemistry_3d_mas.py

from omni.isaac.examples.base_sample import BaseSample
import numpy as np
from omni.isaac.core import World
from omni.isaac.examples.user_examples.LLM.mas_task import Chem_Lab_Task_SL
from omni.isaac.franka import Franka
from omni.physx.scripts import physicsUtils, particleUtils
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
import threading
import os
from omni.isaac.examples.user_examples.LLM.mas import *

class Chemistry3DMAS(BaseSample):
    def __init__(self):
        super().__init__()
        self.my_world = None
        self.Franka = None
        self.controller_manager = None
        self.mas = None
        self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        self.my_dict = {}
        self.input_thread = None

    def setup_scene(self):
        # Create World
        self.my_world = World(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, set_defaults=False)
        self.my_world._physics_context.set_broadphase_type('GPU')
        self.my_world._physics_context.enable_gpu_dynamics(flag=True)
        # Initialize Utils
        self.utils = Utils()
        # Set up the scene
        task = Chem_Lab_Task_SL(name='Chem_Lab_Task_SL')
        self.my_world.add_task(task)

    async def setup_post_load(self):
        # Initialize simulation
        await self.my_world.initialize_simulation_context_async()
        # Reset the world
        await self.my_world.reset_async()
        await self.my_world.pause_async()
        # Get robot and other objects
        self.Franka = self.my_world.scene.get_object("Franka")
        self.my_dict = {
            'Franka': self.Franka,
            'Bottle_Kmno4': self.my_world.scene.get_object("Bottle_Kmno4"),
            'beaker_Kmno4': self.my_world.scene.get_object("beaker_Kmno4"),
            'Bottle_Fecl2': self.my_world.scene.get_object("Bottle_Fecl2"),
            'beaker_Fecl2': self.my_world.scene.get_object("beaker_Fecl2")
        }
        # Set particle parameters
        self.utils._set_particle_parameter(self.my_world, particleContactOffset=0.003)
        # Initialize the agent system
        self.controller_manager = ControllerManager(self.my_world, self.Franka, self.Franka.gripper)
        self.mas = MAS(self.my_world, self.controller_manager)
        # Add particles and containers
        add_particles_str = self.mas._add_particles()
        with open('add_particle_set_str.txt', 'w') as file:
            file.write(add_particles_str)
        self.mas._generate_code_str(add_particles_str)
        self.mas._execute_code_str()
        sim_container_str = self.mas._add_sim_container(add_particles_str)
        with open('add_sim_container_str.txt', 'w') as file:
            file.write(sim_container_str)
        self.mas._generate_code_str(sim_container_str)
        self.mas._execute_code_str()
        add_rigidbody_str = self.mas._add_rigidbody()
        with open('add_rigidbody_str.txt', 'w') as file:
            file.write(add_rigidbody_str)
        self.mas._generate_code_str(add_rigidbody_str)
        self.mas._execute_code_str()
        # Initial user prompt
        user = 'Please observe the test bench and tell me what kind of chemical experiments I can do'
        print(self.mas._response_reaction(user))
        # Start user input thread
        self.user_prompt = None
        self.controllers_ready = False
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        # Register physics callback
        self.my_world.add_physics_callback("sim_step", self.sim_step)

    async def setup_pre_reset(self):
        # Remove physics callback
        self.my_world.remove_physics_callback("sim_step")

    async def setup_post_reset(self):
        # Reset the world
        await self.my_world.reset_async()
        await self.my_world.pause_async()
        # Re-initialize variables
        self.user_prompt = None
        self.controllers_ready = False
        # Start user input thread again
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        # Register physics callback again
        self.my_world.add_physics_callback("sim_step", self.sim_step)

    async def setup_post_clear(self):
        # Clean up
        self.my_world = None
        self.Franka = None
        self.controller_manager = None
        self.mas = None
        self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        self.my_dict = {}

    def get_user_input(self):
        while True:
            self.user_prompt = input("Enter your task: ")

    def sim_step(self, step_size):
        if self.my_world.is_playing():
            if self.my_world.current_time_step_index == 0:
                self.my_world.reset()
                self.controller_manager.reset()
            current_observations = self.my_world.get_observations()
            if self.user_prompt and not self.controllers_ready:
                print('code generating ...')
                # Generate controllers based on the user prompt
                controllers_str = self.mas._generate_controllers(self.user_prompt, current_observations)
                with open('controllers_str.txt', 'w') as file:
                    file.write(controllers_str)
                self.mas._generate_code_str(controllers_str)
                self.mas._execute_code_str()
                add_controllers_str = self.mas._add_controllers(controllers_str)
                with open('add_controllers_str.txt', 'w') as file:
                    file.write(add_controllers_str)
                self.mas._generate_code_str(add_controllers_str)
                self.mas._execute_code_str()
                add_tasks_str = self.mas._add_tasks(add_controllers_str + str(self.my_dict))
                with open('add_tasks_str.txt', 'w') as file:
                    file.write(add_tasks_str)
                self.mas._generate_code_str(add_tasks_str)
                self.mas._execute_code_str()
                self.controllers_ready = True
                print('controllers executing ...')
            if self.controllers_ready:
                # Execute the controller manager
                self.controller_manager.execute(current_observations=current_observations)
                if self.controller_manager.is_done():
                    self.my_world.pause()
                    self.controllers_ready = False  # Reset for next user prompt
                    self.user_prompt = None

    async def on_start_simulation_async(self):
        await self.my_world.play_async()
