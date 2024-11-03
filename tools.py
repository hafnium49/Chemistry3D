# tools.py

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from math import pi

def add_pickmove_task(controller_manager, picking_object, target, current_observations=None, robot=None):
    # Validate picking_object
    valid_objects = ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4']
    if picking_object not in valid_objects:
        return f"Invalid picking_object: {picking_object}. Must be one of {valid_objects}"

    # Get picking_position from observations
    picking_position = current_observations[picking_object]['position']

    # Determine target_position based on the type of 'target'
    if isinstance(target, str):
        # target is a target_object
        target_object = target
        if target_object not in valid_objects:
            return f"Invalid target_object: {target_object}. Must be one of {valid_objects}"
        target_position = current_observations[target_object]['Pour_Position']
    elif isinstance(target, list) or isinstance(target, np.ndarray):
        # target is a target_position
        target_position = np.array(target)
    else:
        return "Error: 'target' must be either a valid object name or a numeric position array."

    # Set default values
    current_joint_positions = robot.get_joint_positions()
    end_effector_offset = np.array([0.0, 0.0, 0.06])
    end_effector_orientation = euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0]))

    param_template = {
        "picking_position": np.array(picking_position),
        "target_position": target_position,
        "current_joint_positions": np.array(current_joint_positions),
        "end_effector_offset": end_effector_offset,
        "end_effector_orientation": end_effector_orientation
    }
    # Store the target_position for use in add_return_task
    controller_manager.last_pickmove_target_position = target_position
    controller_manager.add_task('pickmove_controller', param_template)
    return "PickMove task added successfully."

def add_pour_task(controller_manager, picked_object, current_observations=None, robot=None):
    # # Assume pour_direction is -1 as per assumption
    # pour_direction = -1
    pour_direction = current_observations[picked_object]['Pour_Direction']
    pour_speed = pour_direction * 55 / 180.0 * pi  # Convert 55 degrees to radians

    # # Access robot via controller_manager
    # robot = controller_manager.robot

    # Set default values
    current_joint_positions = robot.get_joint_positions()
    current_joint_velocities = robot.get_joint_velocities()
    franka_art_controller = robot.get_articulation_controller()

    param_template = {
        "franka_art_controller": franka_art_controller,
        "current_joint_positions": np.array(current_joint_positions),
        "current_joint_velocities": np.array(current_joint_velocities),
        "pour_speed": pour_speed
    }
    controller_manager.add_task('pour_controller', param_template)
    return "Pour task added successfully."

def add_return_task(controller_manager, pour_position, return_position, current_observations=None, robot=None):
    # Validate pour_position
    valid_objects = ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4']
    if isinstance(pour_position, str):
        if pour_position not in valid_objects:
            return f"Invalid pour_position: {pour_position}. Must be one of {valid_objects}"
        pour_position_value = current_observations[pour_position]['Pour_Position']
    elif isinstance(pour_position, list) or isinstance(pour_position, np.ndarray):
        pour_position_value = np.array(pour_position)
    else:
        return "Error: 'pour_position' must be either a valid object name or a numeric position array."

    # Validate return_position
    if isinstance(return_position, str):
        if return_position not in valid_objects:
            return f"Invalid return_position: {return_position}. Must be one of {valid_objects}"
        return_position_value = current_observations[return_position]['Return_Position']
    elif isinstance(return_position, list) or isinstance(return_position, np.ndarray):
        return_position_value = np.array(return_position)
    else:
        return "Error: 'return_position' must be either a valid object name or a numeric position array."

    # Set default values
    current_joint_positions = robot.get_joint_positions()
    end_effector_offset = np.array([0.0, 0.0, 0.055])
    end_effector_orientation = euler_angles_to_quat(np.array([np.pi / 2, np.pi / 2, 0]))

    param_template = {
        "pour_position": pour_position_value,
        "return_position": return_position_value,
        "current_joint_positions": np.array(current_joint_positions),
        "end_effector_offset": end_effector_offset,
        "end_effector_orientation": end_effector_orientation
    }
    controller_manager.add_task('return_controller', param_template)
    return "Return task added successfully."

def get_function_schemas():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_pickmove_task",
                "description": (
                    "Adds a pick-and-move task to the controller manager. "
                    "Picks up an object, moves it to a target position and hold."
                    "To release the object, follow this task with an 'add_return_task'."
                    "The 'picking_object' and 'target' define the initial and final positions of the task, respectively."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "picking_object": {
                            "type": "string",
                            "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                            "description": (
                                "The name of the object to pick. This defines the initial position of the task."
                            )
                        },
                        "target": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                                    "description": (
                                        "The name of the target object to move to. This defines the final position of the task."
                                    )
                                },
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": (
                                        "The numeric target position to move the object to. This defines the final position of the task."
                                    )
                                }
                            ],
                            "description": "The target object name or position."
                        }
                    },
                    "required": ["picking_object", "target"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_pour_task",
                "description": (
                    "Adds a pour task to the controller manager. "
                    "This task performs a pouring action at the robot's current position. "
                    "Note: The 'picked_object' must be equivalent to the 'picking_object' of the last step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "picked_object": {
                            "type": "string",
                            "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                            "description": (
                                "The name of the holding object. This defines the pour direction of the task."
                            )
                        },
                    },
                    "required": ["picked_object"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_return_task",
                "description": (
                    "Adds a return task to the controller manager. "
                    "Assumes the robot is currently holding an object and performs a return action to the specified position."
                    "To pick up an object, call 'add_pickmove_task' before this task."
                    "The 'pour_position' and 'return_position' define the initial and final positions of the task, respectively. "
                    "Note: The 'pour_position' must be equivalent to the final position of the last step (e.g., the 'target' of the previous 'add_pickmove_task')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pour_position": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                                    "description": (
                                        "The name of the pour position object. "
                                        "This should be equivalent to the final position of the last task."
                                    )
                                },
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": (
                                        "The numeric pour position. "
                                        "This should be equivalent to the final position of the last task."
                                    )
                                }
                            ],
                            "description": "The pour position as object name or numeric position."
                        },
                        "return_position": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                                    "description": (
                                        "The name of the return position object. "
                                        "This defines the final position of the return task."
                                    )
                                },
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "description": (
                                        "The numeric return position. "
                                        "This defines the final position of the return task."
                                    )
                                }
                            ],
                            "description": "The return position as object name or numeric position."
                        }
                    },
                    "required": ["pour_position", "return_position"],
                    "additionalProperties": False
                }
            }
        }
    ]
    return tools

