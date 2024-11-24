# chemistry_3d_mas.py

import omni.kit.app
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

# Import necessary libraries
import websocket
import json
import numpy as np
import copy  # For deep copying observations
import time  # For time tracking

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
proposed_str_path = os.path.join(current_directory, 'LLM/Proposed_str')

class Chemistry3DMAS(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        # Set world settings
        self._world_settings["physics_dt"] = 1.0 / 120.0
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_prim_path"] = "/physicsScene"
        # self._world_settings["device"] = "cuda" #"cpu" #
        self._world_settings["set_defaults"] = False

        self.controller_manager = None
        self.Franka = None
        self.mas = None
        # self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        # self.input_thread = None
        self.mycamera = None

        # Simulation containers
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None

        # Initialize WebSocket client
        self.ws = None  # WebSocket client
        self.tool_calls_queue = []
        self.websocket_connected = False

        # Initialize previous observations and timing
        self.previous_observations = None
        self.last_observation_time = None  # For limiting print rate

        # Start WebSocket client in a separate thread
        self.websocket_thread = threading.Thread(target=self.start_websocket_client)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()

    # def print_and_send(self, message):
    #     # Print to terminal
    #     print(message)
    #     # Send to WebSocket server if connected
    #     if self.websocket_connected:
    #         try:
    #             self.ws.send(json.dumps({'type': 'log', 'message': str(message)}))
    #         except Exception as e:
    #             print(f'Error sending message to WebSocket server: {e}')

    def print_and_send(self, message):
        # Print to terminal
        print(message)
        # Send to WebSocket server if connected
        if self.websocket_connected:
            try:
                # self.ws.send(json.dumps({
                #     'type': 'conversation.item.create',
                #     'item': {
                #         'type': 'message',
                #         'role': 'user',
                #         'text': str(message)
                #     }
                # }))
                self.ws.send(json.dumps({
                    'type': 'message',
                    'role': 'user',
                    'text': str(message)
                }))

            except Exception as e:
                print(f'Error sending message to WebSocket server: {e}')

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
        self.print_and_send(f"Franka: {self.Franka}")
        if self.Franka is None:
            self.print_and_send("Franka robot not found in the scene.")
        else:
            # Initialize the robot's articulation
            self.Franka.initialize()
            self.Franka.reset_buffers()
            self.print_and_send("Franka robot initialized.")

        self.mycamera = world.scene.get_object("camera")

        # Initialize the controller manager
        self.controller_manager = ControllerManager(world, self.Franka, self.Franka.gripper)
        self.controller_manager.initialize()

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

        # Perform simulation updates
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Bottle_Kmno4, self.Franka, self.controller_manager)
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Beaker_Fecl2, self.Franka, self.controller_manager)  # Added
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Bottle_Fecl2, self.Franka, self.controller_manager)
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Beaker_Kmno4, self.Franka, self.controller_manager)

        # Start user input thread
        # self.user_prompt = None
        self.controllers_ready = False
        # self.input_thread = threading.Thread(target=self.get_user_input)
        # self.input_thread.daemon = True
        # self.input_thread.start()

        # Register physics callback
        world.add_physics_callback("sim_step", self.sim_step)

    async def setup_pre_reset(self):
        world = self.get_world()
        # Remove physics callback
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        # Reset the controller manager
        if self.controller_manager:
            self.controller_manager.reset()

    async def setup_post_reset(self):
        world = self.get_world()
        # Reset the world
        await world.reset_async()
        # Wait for physics steps to ensure the physics simulation view is created
        for _ in range(5):
            await world.step_async()
        # Wait for the stage to load
        await omni.kit.app.get_app().next_update_async()

        # Re-initialize the robot and objects
        self.Franka = world.scene.get_object("Franka")
        self.print_and_send(f"Franka after reset: {self.Franka}")
        if self.Franka is None:
            self.print_and_send("Franka robot not found in the scene after reset.")
        else:
            # Initialize the robot's articulation
            self.Franka.initialize()
            self.Franka.reset_buffers()
            self.print_and_send("Franka robot initialized after reset.")

        self.mycamera = world.scene.get_object("camera")

        # Re-initialize the controller manager after the robot is initialized
        self.controller_manager = ControllerManager(world, self.Franka, self.Franka.gripper)
        self.controller_manager.initialize()

        # Re-initialize simulation containers with specific properties
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

        # Re-initialize the MAS system
        self.mas = MAS(world, self.controller_manager)

        # Perform simulation updates
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Bottle_Kmno4, self.Franka, self.controller_manager)
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Beaker_Fecl2, self.Franka, self.controller_manager)  # Added
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Bottle_Fecl2, self.Franka, self.controller_manager)
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Beaker_Kmno4, self.Franka, self.controller_manager)

        # Re-initialize variables
        self.controllers_ready = False
        self.previous_observations = None
        self.last_observation_time = None

        # Register physics callback again
        world.add_physics_callback("sim_step", self.sim_step)

    async def setup_post_clear(self):
        # Clean up
        self.controller_manager = None
        self.Franka = None
        self.mas = None
        # self.user_prompt = None
        self.controllers_ready = False
        self.utils = None
        # self.input_thread = None
        self.mycamera = None
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None
        self.previous_observations = None
        self.last_observation_time = None

    # def get_user_input(self):
    #     while True:
    #         self.user_prompt = input("Enter your task: ")

    def start_websocket_client(self):
        # Define event handlers
        def on_open(ws):
            print('WebSocket client connected to the relay server')
            self.websocket_connected = True

        def on_message(ws, message):
            print(f'Received message from relay server: {message}')
            try:
                data = json.loads(message)
                if data.get('type') == 'function_call':
                    print(f'Received function_call from relay server: {data}')
                    self.tool_calls_queue.append(data)
                elif data.get('type') == 'log':
                    print(f"Relay server log: {data}")
            except json.JSONDecodeError as e:
                print(f'Error decoding message: {e}')

        def on_error(ws, error):
            print(f'WebSocket error: {error}')

        def on_close(ws):
            print('WebSocket client disconnected from the relay server')
            self.websocket_connected = False

        # Create WebSocket app
        ws_url = 'ws://localhost:8081/chemistry3d'  # Ensure the URL matches the relay server
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run the WebSocket client
        self.ws.run_forever()

    def observations_changed(self, obs1, obs2):
        # Check if observations have changed
        if obs1.keys() != obs2.keys():
            return True
        for key in obs1:
            val1 = obs1[key]
            val2 = obs2[key]
            if isinstance(val1, dict) and isinstance(val2, dict):
                if self.observations_changed(val1, val2):
                    return True
            elif isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if not np.array_equal(val1, val2):
                    return True
            else:
                if val1 != val2:
                    return True
        return False

    def sim_step(self, step_size):
        world = self.get_world()
        if world.is_playing():
            # Added check to ensure physics simulation view is created
            if not self.Franka.is_simulation_view_created():
                # Skip this step until the simulation view is ready
                return

            current_observations = world.get_observations()
            current_time = time.time()

            # Check if observations have changed and limit print rate to 1 Hz
            if (self.previous_observations is None) and (self.last_observation_time is None or (current_time - self.last_observation_time) >= 1.0):
            # if (self.previous_observations is None or self.observations_changed(self.previous_observations, current_observations)) and (self.last_observation_time is None or (current_time - self.last_observation_time) >= 1.0):
                self.print_and_send(f"Current Observations: {current_observations}")
                self.previous_observations = copy.deepcopy(current_observations)
                self.last_observation_time = current_time

            # if self.user_prompt and not self.controllers_ready:
            #     # Send user prompt to the relay server
            #     if self.websocket_connected:
            #         self.print_and_send(f'Sending user prompt to relay server: {self.user_prompt}')
            #         message = json.dumps({
            #             'type': 'message',
            #             'text': self.user_prompt
            #         })
            #         # self.ws.send(message)
            #         self.user_prompt = None  # Reset user prompt after sending
            #     else:
            #         self.print_and_send('WebSocket is not connected. Cannot send user input.')

            # Check if there are any function calls received
            if self.tool_calls_queue:
                function_call = self.tool_calls_queue.pop(0)
                try:
                    self.print_and_send(f"Processing function call: {function_call}")
                    # Handle the function call
                    result = self.mas.agent_assistant.handle_function_call(
                        tool_call=function_call,
                        global_dict=globals(),
                        controller_manager=self.controller_manager,
                        current_observations=current_observations,
                        robot=self.Franka
                    )
                    self.print_and_send(f"Function call result: {result}")
                    # Send function call output back to the relay server
                    output_data = {
                        'type': 'function_call_output',
                        'call_id': function_call.get('id', ''),
                        'output': result
                    }
                    self.ws.send(json.dumps(output_data))
                    self.controllers_ready = True
                    self.print_and_send('Controllers executing...')
                except Exception as e:
                    self.print_and_send(f"Error processing function_call: {e}")

            if self.controllers_ready:
                # Execute the controller manager
                self.controller_manager.execute(current_observations=current_observations)
                if self.controller_manager.is_done():
                    world.pause()
                    self.controllers_ready = False  # Reset for next user prompt
                    # self.user_prompt = None

    async def on_start_simulation_async(self):
        world = self.get_world()
        await world.play_async()
