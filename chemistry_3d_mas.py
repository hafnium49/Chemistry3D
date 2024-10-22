# chemistry_3d_mas.py

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from mas_task import Chem_Lab_Task_SL
from omni.isaac.franka import Franka
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Sdf, Gf, UsdPhysics
from omni.isaac.sensor import Camera
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.physx.scripts import physicsUtils, particleUtils
from omni.isaac.examples.user_examples.Controllers.Controller_Manager import ControllerManager
from omni.isaac.examples.user_examples.Controllers.pick_move_controller import PickMoveController
from omni.isaac.examples.user_examples.Controllers.pour_controller import PourController
from omni.isaac.examples.user_examples.Controllers.return_controller import PlaceController as ReturnController
from omni.isaac.examples.user_examples.Sim_Container import Sim_Container
from omni.isaac.examples.user_examples.Chemistry3D_utils import Utils
import logging
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from mas import *
import threading

class Chemistry3DMAS:
    def __init__(self):
        self.my_world = World(physics_dt=1.0 / 120.0, stage_units_in_meters=1.0, set_defaults=False)
        self.my_world._physics_context.set_broadphase_type('GPU')
        self.my_world._physics_context.enable_gpu_dynamics(flag=True)
        self.utils = Utils()
        self.stage = self.my_world.scene.stage
        self.scenePath = Sdf.Path("/physicsScene")
        self.task = Chem_Lab_Task_SL(name='Chem_Lab_Task_SL')
        self.my_world.add_task(self.task)
        self.my_world.reset()

        self.Franka = self.my_world.scene.get_object("Franka")

        self.Bottle_Kmno4 = self.my_world.scene.get_object("Bottle_Kmno4")
        self.Bottle_Fecl2 = self.my_world.scene.get_object("Bottle_Fecl2")
        self.beaker_Kmno4 = self.my_world.scene.get_object("beaker_Kmno4")
        self.beaker_Fecl2 = self.my_world.scene.get_object("beaker_Fecl2")

        self.my_dict = {
            'Franka': self.my_world.scene.get_object("Franka"),
            'Bottle_Kmno4': self.my_world.scene.get_object("Bottle_Kmno4"),
            'beaker_Kmno4': self.my_world.scene.get_object("beaker_Kmno4"),
            'Bottle_Fecl2': self.my_world.scene.get_object("Bottle_Fecl2"),
            'beaker_Fecl2': self.my_world.scene.get_object("beaker_Fecl2")
        }

        self.current_observations = self.my_world.get_observations()
        self.utils._set_particle_parameter(self.my_world, particleContactOffset=0.003)
        self.controller_manager = ControllerManager(self.my_world, self.Franka, self.Franka.gripper)
        self.mas = MAS(self.my_world, self.controller_manager)

        self.prepare_simulation()

        self.user_prompt = None
        self.controllers_ready = False

    def prepare_simulation(self):
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

        user = 'Please observe the test bench and tell me what kind of chemical experiments I can do'
        print(self.mas._response_reaction(user))

    def get_user_input(self):
        while True:
            self.user_prompt = input("Enter your task: ")

    def run(self):
        input_thread = threading.Thread(target=self.get_user_input)
        input_thread.daemon = True
        input_thread.start()

        while simulation_app.is_running():
            self.my_world.step(render=True)
            if self.my_world.is_playing():
                if self.my_world.current_time_step_index == 0:
                    self.my_world.reset()
                    self.controller_manager.reset()

                self.current_observations = self.my_world.get_observations()

                if self.user_prompt and not self.controllers_ready:
                    print('Code generating ...')
                    # Generate controllers based on the user prompt
                    controllers_str = self.mas._generate_controllers(self.user_prompt, self.current_observations)
                    with open('controllers_str.txt', 'w') as file:
                        file.write(controllers_str)
                    with open('controllers_str.txt', 'r') as f:
                        controllers_str = f.read()
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
                    print('Controllers executing ...')

                if self.controllers_ready:
                    # Execute the controller manager
                    self.controller_manager.execute(current_observations=self.current_observations)

                    if self.controller_manager.is_done():
                        self.my_world.pause()
                        self.controllers_ready = False  # Reset for next user prompt
                        self.user_prompt = None

        simulation_app.close()
