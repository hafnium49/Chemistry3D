# Launch Isaac Sim before any other imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Sdf, UsdPhysics, PhysxSchema
import omni.usd

from Chemistry3D_Task import Chem_Lab_Task
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils

print("complete omniverse imports")

import sys
from Chemistry3D_utils import Utils
from Controllers.Controller_Manager import ControllerManager
from Sim_Container import Sim_Container
import logging
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Initialize the simulation context with GPU backend and physics time step
simulation_context = SimulationContext(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, backend="torch", device="cuda", set_defaults=False)

# Enable the PhysX extension for GPU dynamics
enable_extension("omni.physx")

# Get the stage
stage = omni.usd.get_context().get_stage()

# Create a physics scene if it doesn't exist and enable GPU dynamics
scenePath = Sdf.Path("/physicsScene")
physics_scene = stage.GetPrimAtPath(scenePath)
if not physics_scene.IsValid():
    physicsScene = UsdPhysics.Scene.Define(stage, scenePath)

# Enable GPU dynamics for the physics scene
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene)
physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

# Import utility class to set particle parameters
utils = Utils()
utils._set_particle_parameter(simulation_context, particleContactOffset=0.003)

# Add the chemical lab task to the simulation world
chem_lab_task = Chem_Lab_Task(name='Chem_Lab_Task')
simulation_context.add_task(chem_lab_task)
simulation_context.reset()

# Retrieve objects from the scene
Franka0 = simulation_context.scene.get_object("Franka0")
mycamera = simulation_context.scene.get_object("camera")
current_observations = simulation_context.get_observations()
controller_manager = ControllerManager(simulation_context, Franka0, Franka0.gripper)

# Initialize simulation containers with specific properties
Sim_Bottle1 = Sim_Container(world=simulation_context, sim_container=simulation_context.scene.get_object("Bottle1"),
                            solute={'MnO4^-': 0.02, 'K^+': 0.02, 'H^+': 0.04, 'SO4^2-': 0.02}, volume=0.02)
Sim_Bottle2 = Sim_Container(world=simulation_context, sim_container=simulation_context.scene.get_object("Bottle2"),
                            solute={'Fe^2+': 0.06, 'Cl^-': 0.12}, volume=0.02)
Sim_Beaker1 = Sim_Container(world=simulation_context, sim_container=simulation_context.scene.get_object("Beaker1"))
Sim_Beaker2 = Sim_Container(world=simulation_context, sim_container=simulation_context.scene.get_object("Beaker2"))

Sim_Beaker1.sim_update(Sim_Bottle1, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Bottle2, Franka0, controller_manager)
Sim_Beaker2.sim_update(Sim_Beaker1, Franka0, controller_manager)

# Main simulation loop
while simulation_app.is_running():
    simulation_context.step(render=True)
    if simulation_context.is_playing():
        if simulation_context.current_time_step_index == 0:
            simulation_context.reset()
            controller_manager.reset()
        current_observations = simulation_context.get_observations()
        controller_manager.execute(current_observations=current_observations)
        controller_manager.process_concentration_iters()
        if controller_manager.need_new_liquid():
            controller_manager.get_current_controller()._get_sim_container2().create_liquid(controller_manager, current_observations)
        if controller_manager.is_done():
            simulation_context.pause()
            break

# Close the simulation application
simulation_app.close()
