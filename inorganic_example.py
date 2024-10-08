# Import the 'isaacsim' module first
import isaacsim

# Import SimulationApp directly from 'isaacsim'
from isaacsim import SimulationApp

# Initialize the simulation application with VR and required extensions enabled
config = {
    "headless": False,
    "extensions": {
        # Core VR extensions
        "omni.kit.xr.core": {"enabled": True},
        "omni.kit.xr.profile.common": {"enabled": True},
        "omni.kit.xr.profile.vr": {"enabled": True},
        "omni.kit.xr.system.openxr": {"enabled": True},
        "omni.kit.xr.ui.config.metaquest": {"enabled": True},
        "omni.kit.xr.ui.config.common": {"enabled": True},
        "omni.kit.xr.ui.stage.common": {"enabled": True},
        "omni.kit.xr.ui.window.profile": {"enabled": True},
        "omni.kit.xr.ui.window.viewport": {"enabled": True},
        "omni.kit.xr.advertise": {"enabled": True},
        # Required services extensions that must be loaded at startup
        "omni.services.facilities.base": {"enabled": True},
        "omni.services.core": {"enabled": True},
        "omni.services.transport.server.base": {"enabled": True},
        "omni.services.transport.server.zeroconf": {"enabled": True},
    },
}

simulation_app = SimulationApp(config)

# Rest of your imports and code
import numpy as np
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage

from Chemistry3D_Task import Chem_Lab_Task
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Gf, UsdPhysics, PhysxSchema
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
import omni.usd

print("complete omniverse imports")

from Chemistry3D_utils import Utils  # Import local utils.py

from Controllers.Controller_Manager import ControllerManager
from Sim_Container import Sim_Container

import logging
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Initialize the simulation world with GPU dynamics enabled
my_world = World(
    physics_dt=1.0 / 120.0,
    stage_units_in_meters=1.0,
    physics_prim_path="/physicsScene",
    device="cuda",  # Use 'gpu' (case-insensitive)
    set_defaults=False,
)

# Get the physics context and enable GPU dynamics
physics_context = my_world.get_physics_context()
physics_context.enable_gpu_dynamics(True)

# Get the stage
stage = my_world.scene.stage

# Ensure the physics scene exists at '/physicsScene'
scene_path = Sdf.Path("/physicsScene")
if not stage.GetPrimAtPath(scene_path):
    physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
else:
    physics_scene = UsdPhysics.Scene(stage.GetPrimAtPath(scene_path))

# Apply PhysxSceneAPI and set the enableGPUDynamics attribute to True
physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
if physics_scene_prim.IsValid():
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_scene_api.CreateEnableGPUDynamicsAttr().Set(True)

# Initialize utils and set particle parameters
utils = Utils()
utils._set_particle_parameter(my_world, particleContactOffset=0.003)

# Add the chemical lab task to the simulation world
my_world.add_task(Chem_Lab_Task(name='Chem_Lab_Task'))
my_world.reset()

# Retrieve objects from the scene
Franka0 = my_world.scene.get_object("Franka0")
mycamera = my_world.scene.get_object("camera")
current_observations = my_world.get_observations()
controller_manager = ControllerManager(my_world, Franka0, Franka0.gripper)

# Initialize simulation containers with specific properties
Sim_Bottle1 = Sim_Container(
    world=my_world,
    sim_container=my_world.scene.get_object("Bottle1"),
    solute={'MnO4^-': 0.02, 'K^+': 0.02, 'H^+': 0.04, 'SO4^2-': 0.02},
    volume=0.02
)
Sim_Bottle2 = Sim_Container(
    world=my_world,
    sim_container=my_world.scene.get_object("Bottle2"),
    solute={'Fe^2+': 0.06, 'Cl^-': 0.12},
    volume=0.02
)
Sim_Beaker1 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Beaker1"))
Sim_Beaker2 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Beaker2"))

Sim_Beaker1.sim_update(Sim_Bottle1, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Bottle2, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Beaker1, Franka0, controller_manager)

# Configure the VR interface
import omni.kit.xr

# Get the VR interface
xr_interface = omni.kit.xr.get_xr_interface()

# Start the VR system
xr_interface.startup()

# Set the viewport window for VR rendering (ensure the viewport name matches your setup)
xr_interface.set_viewport_window("Viewport")

# Optional: Configure VR-specific settings
xr_interface.set_eye_resolution(1920, 1080)  # Adjust based on your device capabilities
xr_interface.set_refresh_rate(90)  # Adjust based on your device capabilities

# Main simulation loop
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            controller_manager.reset()
        current_observations = my_world.get_observations()
        controller_manager.execute(current_observations=current_observations)
        controller_manager.process_concentration_iters()
        if controller_manager.need_new_liquid():
            controller_manager.get_current_controller()._get_sim_container2().create_liquid(controller_manager, current_observations)
        if controller_manager.is_done():
            my_world.pause()
            break

# Close the simulation application
simulation_app.close()
