# Launch Isaac Sim before any other imports
# These are the default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from Chemistry3D_Task import Chem_Lab_Task
# from Chemistry3D_Demo.Chemistry3D_Task import Chem_Lab_Task
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Gf, UsdPhysics
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils

from Chemistry3D_utils import Utils  # Import local utils.py
from Controllers.Controller_Manager import ControllerManager
from Sim_Container import Sim_Container
# from utils import Utils
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Initialize the simulation world
my_world = World(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, set_defaults=False)
my_world._physics_context.enable_gpu_dynamics(flag=True)
stage = my_world.scene.stage
scenePath = Sdf.Path("/physicsScene")
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

Sim_Bottle1 = Sim_Container(world = my_world, sim_container = my_world.scene.get_object("Bottle1"), solute={'H^+': 30, 'Cl^-': 30}, volume=10)
Sim_Beaker1 = Sim_Container(world = my_world,sim_container = my_world.scene.get_object("Beaker1"))
Sim_Beaker2 = Sim_Container(world = my_world,sim_container = my_world.scene.get_object("Beaker2"),solute={'FeO': 2}, volume='s')

Sim_Beaker1.sim_update(Sim_Bottle1,Franka0,controller_manager)
Sim_Beaker2.sim_update(Sim_Beaker1,Franka0,controller_manager)

# count = 0
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
        # if count % 10 == 0:
        #     img = mycamera.get_rgba()
        #     file_name = os.path.join(root_path, f"{count}")
        #     save_rgb(img, file_name)
        # count += 1
        if controller_manager.is_done():
            my_world.pause()
            break

# Close the simulation application
simulation_app.close()
