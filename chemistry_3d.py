# chemistry_3d.py

import numpy as np
import omni  # Ensure this import is present
from omni.isaac.core import World
from omni.isaac.examples.base_sample import BaseSample
from pxr import Sdf, UsdPhysics, PhysxSchema

# Import local modules
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
from omni.isaac.examples.user_examples.Chemistry3D_Task import Chem_Lab_Task
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Sim_Container import Sim_Container

class Chemistry3D(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["physics_dt"] = 1.0 / 120.0
        # self._world_settings["rendering_dt"] = 1.0 / 60.0
        self._world_settings["physics_prim_path"] = "/physicsScene"
        # self._world_settings["device"] = "cpu" #"cuda"
        self._world_settings["set_defaults"] = False
        # self._world_settings["backend"] = "torch"  # Add this line to set the backend to PyTorch
        self.controller_manager = None
        self.Franka0 = None
        self.mycamera = None
        self.utils = None
        self.Sim_Bottle1 = None
        self.Sim_Bottle2 = None
        self.Sim_Beaker1 = None
        self.Sim_Beaker2 = None

    def setup_scene(self):
        world = self.get_world()
        # world.initialize()  # Not needed as the world is initialized in BaseSample

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

        # Add the chemical lab task to the simulation world
        world.add_task(Chem_Lab_Task(name='Chem_Lab_Task'))

    async def setup_post_load(self):
        world = self.get_world()
        # Wait for the stage to load
        await omni.kit.app.get_app().next_update_async()

        # Get the robot and initialize it
        self.Franka0 = world.scene.get_object("Franka0")
        print(f"Franka0: {self.Franka0}")
        if self.Franka0 is None:
            print("Franka0 robot not found in the scene.")
        # else:
        #     self.Franka0.initialize()
        #     await self.Franka0.wait_for_loading_async()
        self.mycamera = world.scene.get_object("camera")

        # Initialize the controller manager
        self.controller_manager = ControllerManager(world, self.Franka0, self.Franka0.gripper)

        # Initialize simulation containers with specific properties
        self.Sim_Bottle1 = Sim_Container(
            world=world,
            sim_container=world.scene.get_object("Bottle1"),
            solute={'MnO4^-': 0.02, 'K^+': 0.02, 'H^+': 0.04, 'SO4^2-': 0.02},
            volume=0.02
        )
        self.Sim_Bottle2 = Sim_Container(
            world=world,
            sim_container=world.scene.get_object("Bottle2"),
            solute={'Fe^2+': 0.06, 'Cl^-': 0.12},
            volume=0.02
        )
        self.Sim_Beaker1 = Sim_Container(world=world, sim_container=world.scene.get_object("Beaker1"))
        self.Sim_Beaker2 = Sim_Container(world=world, sim_container=world.scene.get_object("Beaker2"))

        # Perform simulation updates
        self.Sim_Beaker1.sim_update(self.Sim_Bottle1, self.Franka0, self.controller_manager)
        self.Sim_Beaker2.sim_update(self.Sim_Bottle2, self.Franka0, self.controller_manager)
        self.Sim_Beaker2.sim_update(self.Sim_Beaker1, self.Franka0, self.controller_manager)

    def _on_simulation_step(self, step_size):
        world = self.get_world()
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                self.controller_manager.reset()
            current_observations = world.get_observations()
            self.controller_manager.execute(current_observations=current_observations)
            self.controller_manager.process_concentration_iters()
            if self.controller_manager.need_new_liquid():
                self.controller_manager.get_current_controller()._get_sim_container2().create_liquid(self.controller_manager, current_observations)
            if self.controller_manager.is_done():
                world.pause()

    async def on_start_simulation_async(self):
        world = self.get_world()
        world.add_physics_callback("sim_step", self._on_simulation_step)
        await world.play_async()
        return

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        if self.controller_manager:
            self.controller_manager.reset()
        return

    def world_cleanup(self):
        self.controller_manager = None
        self.Franka0 = None
        self.mycamera = None
        self.utils = None
        self.Sim_Bottle1 = None
        self.Sim_Bottle2 = None
        self.Sim_Beaker1 = None
        self.Sim_Beaker2 = None
        return
