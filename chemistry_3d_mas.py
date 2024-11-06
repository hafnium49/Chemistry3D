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

# Additional imports for WebSocket
import asyncio
import websockets  # Ensure this package is installed: pip install websockets
import json
import threading

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
proposed_str_path = os.path.join(current_directory, 'LLM/Proposed_str')

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
        self.controllers_ready = False
        self.utils = None
        self.mycamera = None

        # Simulation containers
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None

        # WebSocket setup
        self.tool_calls = None
        self.tool_calls_lock = threading.Lock()

        # Initialize an asyncio event loop
        self.loop = asyncio.get_event_loop()
        self.websocket_server = None  # Will hold the server object

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

        # Perform simulation updates
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Bottle_Kmno4, self.Franka, self.controller_manager)
        self.Sim_Beaker_Kmno4.sim_update(self.Sim_Beaker_Fecl2, self.Franka, self.controller_manager)  # Added
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Bottle_Fecl2, self.Franka, self.controller_manager)
        self.Sim_Beaker_Fecl2.sim_update(self.Sim_Beaker_Kmno4, self.Franka, self.controller_manager)

        # Start the WebSocket server
        await self.start_websocket_server()

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
        # Stop the WebSocket server
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            self.websocket_server = None

    async def setup_post_reset(self):
        world = self.get_world()
        # Reset the world
        await world.reset_async()
        await world.pause_async()
        # Wait for the stage to load
        await omni.kit.app.get_app().next_update_async()

        # Re-initialize the robot and objects
        self.Franka = world.scene.get_object("Franka")
        print(f"Franka after reset: {self.Franka}")
        if self.Franka is None:
            print("Franka robot not found in the scene after reset.")
        self.mycamera = world.scene.get_object("camera")

        # Re-initialize the controller manager
        self.controller_manager = ControllerManager(world, self.Franka, self.Franka.gripper)

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

        # Start the WebSocket server again
        await self.start_websocket_server()

        # Register physics callback again
        world.add_physics_callback("sim_step", self.sim_step)

    async def setup_post_clear(self):
        # Clean up
        self.controller_manager = None
        self.Franka = None
        self.mas = None
        self.controllers_ready = False
        self.utils = None
        self.mycamera = None
        self.Sim_Bottle_Kmno4 = None
        self.Sim_Bottle_Fecl2 = None
        self.Sim_Beaker_Kmno4 = None
        self.Sim_Beaker_Fecl2 = None
        self.tool_calls = None
        self.tool_calls_lock = None
        # Stop the WebSocket server if it's running
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
            self.websocket_server = None

    # Modify run_websocket_server to be compatible with Isaac Sim's event loop
    async def start_websocket_server(self):
        async def handler(websocket, path):
            print(f"Client connected from {websocket.remote_address}")
            try:
                async for message in websocket:
                    # Process the message
                    print(f"Received message: {message}")
                    try:
                        # Assume message is a JSON string containing tool_calls
                        tool_calls = json.loads(message)
                        with self.tool_calls_lock:
                            self.tool_calls = tool_calls
                        # Optionally send a response back
                        response = "Message received and processed"
                        await websocket.send(response)
                        print(f"Sent response: {response}")
                    except json.JSONDecodeError as e:
                        error_msg = f"JSON decode error: {e}"
                        print(error_msg)
                        await websocket.send(error_msg)
            except websockets.ConnectionClosed as e:
                print(f"Client disconnected: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            finally:
                print(f"Connection with client {websocket.remote_address} closed")

        # Start the server and keep a reference to it
        print("Starting WebSocket server...")
        self.websocket_server = await websockets.serve(handler, 'localhost', 8765)
        print("WebSocket server started and listening on ws://localhost:8765")

    def sim_step(self, step_size):
        world = self.get_world()
        if world.is_playing():
            if world.current_time_step_index == 0:
                world.reset()
                self.controller_manager.reset()
            current_observations = world.get_observations()
            if not self.controllers_ready:
                with self.tool_calls_lock:
                    if self.tool_calls:
                        tool_calls = self.tool_calls
                        self.tool_calls = None  # Reset for next time
                    else:
                        tool_calls = None
                if tool_calls:
                    print(f"Processing tool_calls: {tool_calls}")
                    try:
                        # Process the tool_calls
                        for k, tool_call in enumerate(tool_calls):
                            print(f"Processing tool_call {k+1}")
                            result = self.mas.agent_assistant.handle_function_call(
                                tool_call=tool_call,
                                global_dict=globals(),
                                controller_manager=self.controller_manager,
                                current_observations=current_observations,
                                robot=self.Franka
                            )
                            print(f"Function call result: {result}")
                        self.controllers_ready = True
                        print('Controllers executing...')
                    except Exception as e:
                        print(f"Error processing tool_calls: {e}")
            if self.controllers_ready:
                # Execute the controller manager
                self.controller_manager.execute(current_observations=current_observations)
                if self.controller_manager.is_done():
                    world.pause()
                    self.controllers_ready = False  # Reset for next tool_calls

    async def on_start_simulation_async(self):
        world = self.get_world()
        await world.play_async()
