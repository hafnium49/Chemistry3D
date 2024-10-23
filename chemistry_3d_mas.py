# chemistry_3d_mas.py

import omni  # Ensure this import is present
from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core import World
from omni.isaac.examples.user_examples.LLM.mas_task import Chem_Lab_Task_SL
from omni.isaac.franka import Franka
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
import threading
import os
from omni.isaac.examples.user_examples.LLM.mas import MAS
from omni.isaac.examples.user_examples.Sim_Container import Sim_Container
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
        self.input_thread = None
        self.mycamera = None

        # Simulation containers
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None

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
        # Wait for the stage to load
        await omni.kit.app.get_app().next_update_async()

        # Get the robot and initialize it
        self.Franka = world.scene.get_object("Franka")
        print(f"Franka: {self.Franka}")
        if self.Franka is None:
            print("Franka robot not found in the scene.")
        self.mycamera = world.scene.get_object("camera")

        # Initialize the controller manager
        self.controller_manager = ControllerManager(world, self.Franka, self.Franka.gripper)

        # Initialize simulation containers with specific properties
        self.Sim_Bottle_Kmno4 = Sim_Container(
            world=world,
            sim_container=world.scene.get_object("Bottle_Kmno4"),
            solute={'MnO4^-': 0.02, 'K^+': 0.02, 'H^+': 0.04, 'SO4^2-': 0.02},
            volume=0.02
        )
        self.Sim_Bottle_Fecl2 = Sim_Container(
            world=world,
            sim_container=world.scene.get_object("Bottle_Fecl2"),
            solute={'Fe^2+': 0.06, 'Cl^-': 0.12},
            volume=0.02
        )
        self.Sim_Beaker_Kmno4 = Sim_Container(world=world, sim_container=world.scene.get_object("beaker_Kmno4"))
        self.Sim_Beaker_Fecl2 = Sim_Container(world=world, sim_container=world.scene.get_object("beaker_Fecl2"))

        # Initialize the MAS system
        self.mas = MAS(world, self.controller_manager)

        # Perform simulation updates (if needed)
        # For this MAS implementation, we may rely on dynamic code generation
        # Alternatively, we can directly call sim_update methods
        # Here, we will proceed with direct calls as per chemistry_3d.py

        # Perform simulation updates
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Bottle_Kmno4, self.Franka, self.controller_manager)
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Bottle_Fecl2, self.Franka, self.controller_manager)
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Beaker_Kmno4, self.Franka, self.controller_manager)

        # Start user input thread
        self.user_prompt = None
        self.controllers_ready = False
        self.input_thread = threading.Thread(target=self.get_user_input)
        self.input_thread.daemon = True
        self.input_thread.start()

        # Register physics callback
        world.add_physics_callback("sim_step", self.sim_step)

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
                print('Code generating...')
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
                add_tasks_str = self.mas._add_tasks(add_controllers_str)
                with open('add_tasks_str.txt', 'w') as file:
                    file.write(add_tasks_str)
                self.mas._generate_code_str(add_tasks_str)
                self.mas._execute_code_str()
                self.controllers_ready = True
                print('Controllers executing...')
            if self.controllers_ready:
                # Execute the controller manager
                self.controller_manager.execute(current_observations=current_observations)
                if self.controller_manager.is_done():
                    world.pause()
                    self.controllers_ready = False  # Reset for next user prompt
                    self.user_prompt = None

    async def setup_pre_reset(self):
        world = self.get_world()
        # Remove physics callback
        if world.physics_callback_exists("sim_step"):
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
        self.input_thread = None
        self.mycamera = None
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None

    async def on_start_simulation_async(self):
        world = self.get_world()
        await world.play_async()
