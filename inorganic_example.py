# Launch Isaac Sim before any other imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World, SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
# from omni.isaac.core.utils.extensions import enable_extension

from Chemistry3D_Task import Chem_Lab_Task
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Gf, UsdPhysics
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
from pxr import PhysxSchema
import omni.usd

print("complete omniverse imports")

import sys

# Import utils explicitly from the local directory
from Chemistry3D_utils import Utils  # Import local utils.py

from Controllers.Controller_Manager import ControllerManager
from Sim_Container import Sim_Container

import logging
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Initialize the simulation world
my_world = World(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, set_defaults=False)
my_world._physics_context.enable_gpu_dynamics(flag=True)
stage = my_world.scene.stage
scenePath = Sdf.Path("/physicsScene")

# # Initialize the simulation world with a specific physics_dt
# my_world = World(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, set_defaults=False)

# # Enable GPU dynamics for the physics context
# physics_context = my_world.get_physics_context()
# physics_context.enable_gpu_dynamics(True)

# # Enable the PhysX extension
# enable_extension("omni.physx")

# # Get the stage
# stage = my_world.scene.stage

# # Create a physics scene if it doesn't exist and enable GPU dynamics
# scenePath = Sdf.Path("/physicsScene")
# if not stage.GetPrimAtPath(scenePath):
#     physicsScene = UsdPhysics.Scene.Define(stage, scenePath)
#     physicsScene.CreateEnableGPUDynamicsAttr(True)

# # Access the physics scene directly through PhysxSchema
# physics_scene = stage.GetPrimAtPath("/physicsScene")
# if physics_scene.IsValid():
#     physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
#     physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

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
Sim_Bottle1 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Bottle1"),
                            solute={'MnO4^-': 0.02, 'K^+': 0.02, 'H^+': 0.04, 'SO4^2-': 0.02}, volume=0.02)
Sim_Bottle2 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Bottle2"),
                            solute={'Fe^2+': 0.06, 'Cl^-': 0.12}, volume=0.02)
Sim_Beaker1 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Beaker1"))
Sim_Beaker2 = Sim_Container(world=my_world, sim_container=my_world.scene.get_object("Beaker2"))

Sim_Beaker1.sim_update(Sim_Bottle1, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Bottle2, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Beaker1, Franka0, controller_manager)

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
