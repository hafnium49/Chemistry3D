# chemistry_3d_mas.py

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core import World
from omni.isaac.examples.user_examples.LLM.mas_task import Chem_Lab_Task_SL
from omni.isaac.franka import Franka
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
import threading
import os
from omni.isaac.examples.user_examples.LLM.mas import MAS
from pxr import Sdf, UsdPhysics, PhysxSchema

class Chemistry3DMAS(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        # Set world settings
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_prim_path"] = "/physicsScene"
        self._world_settings["set_defaults"] = False

        self.controller_manager = None
        self.Franka = None
        self.mas = None
        self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        self.my_dict = {}
        self.input_thread = None

    def setup_scene(self):
        world = self.get_world()
        # Enable GPU dynamics
        physics_context = world.get_physics_context()
        physics_context.enable_gpu_dynamics(True)

        # Ensure the physics scene exists
        stage = world.scene.stage
        scene_path = Sdf.Path("/physicsScene")
        if not stage.GetPrimAtPath(scene_path):
            physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
        else:
            physics_scene = UsdPhysics.Scene(stage.GetPrimAtPath(scene_path))

        physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
        if physics_scene_prim.IsValid():
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
            physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)

        # Initialize utils and set particle parameters
        self.utils = Utils()
        self.utils._set_particle_parameter(world, particleContactOffset=0.003)

        # Set up the scene
        task = Chem_Lab_Task_SL(name='Chem_Lab_Task_SL')
        world.add_task(task)

    async def setup_post_load(self):
        world = self.get_world()
        # Initialize simulation
        await world.initialize_simulation_context_async()
        # Reset the world
        await world.reset_async()
        await world.pause_async()
        # Get robot and other objects
        self.Franka = world.scene.get_object("Franka")
        self.my_dict = {
            'Franka': self.Franka,
            'Bottle_Kmno4': world.scene.get_object("Bottle_Kmno4"),
            'beaker_Kmno4': world.scene.get_object("beaker_Kmno4"),
            'Bottle_Fecl2': world.scene.get_object("Bottle_Fecl2"),
            'beaker_Fecl2': world.scene.get_object("beaker_Fecl2")
        }
        # Set particle parameters
        self.utils._set_particle_parameter(world, particleContactOffset=0.003)
        # Initialize the agent system
        self.controller_manager = ControllerManager(world, self.Franka, self.Franka.gripper)
        self.mas = MAS(world, self.controller_manager)
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
        world.add_physics_callback("sim_step", self.sim_step)

    async def setup_pre_reset(self):
        world = self.get_world()
        # Remove physics callback
        world.remove_physics_callback("sim_step")

    async def setup_post_reset(self):
        world = self.get_world()
        # Reset the world
        await world.reset_async()
        await world.pause_async()
        # Re-initialize variables
        self.user_prompt = None
        self.controllers_ready = False
        # Start user input thread again
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True
        self.input_thread.start()
        # Register physics callback again
        world.add_physics_callback("sim_step", self.sim_step)

    async def setup_post_clear(self):
        # Clean up
        self.controller_manager = None
        self.Franka = None
        self.mas = None
        self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        self.my_dict = {}
        self.input_thread = None

    def get_user_input(self):
        while True:
            self.user_prompt = input("Enter your task: ")

    def sim_step(self, step_size):
        world = self.get_world()
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                self.controller_manager.reset()
            current_observations = world.get_observations()
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
                    world.pause()
                    self.controllers_ready = False  # Reset for next user prompt
                    self.user_prompt = None

    async def on_start_simulation_async(self):
        world = self.get_world()
        await world.play_async()
