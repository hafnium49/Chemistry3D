# tools.py

import numpy as np

def add_pickmove_task(controller_manager, picking_object, target_object=None, target_position=None, current_joint_positions=None, end_effector_offset=None, end_effector_orientation=None, current_observations=None, robot=None):
    # Validate picking_object
    valid_objects = ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4']
    if picking_object not in valid_objects:
        return f"Invalid picking_object: {picking_object}. Must be one of {valid_objects}"

    # Get picking_position from observations
    picking_position = current_observations[picking_object]['position']

    # Determine target_position
    if target_position is not None:
        target_position = np.array(target_position)
    elif target_object is not None:
        if target_object not in valid_objects:
            return f"Invalid target_object: {target_object}. Must be one of {valid_objects}"
        target_position = current_observations[target_object]['Pour_Position']
    else:
        return "Error: Either target_position or target_object must be provided."

    # Set default values
    if current_joint_positions is None:
        current_joint_positions = robot.get_joint_positions()
    if end_effector_offset is None:
        end_effector_offset = np.array([0.0, 0.0, 0.06])
    if end_effector_orientation is None:
        end_effector_orientation = [1.0, 0.0, 0.0, 0.0]  # Default quaternion orientation

    param_template = {
        "picking_position": np.array(picking_position),
        "target_position": target_position,
        "current_joint_positions": np.array(current_joint_positions),
        "end_effector_offset": end_effector_offset,
        "end_effector_orientation": end_effector_orientation
    }
    controller_manager.add_task('pickmove_controller', param_template)
    return "PickMove task added successfully."

def add_pour_task(controller_manager, pour_speed, current_joint_positions=None, current_joint_velocities=None, current_observations=None, robot=None):
    # Set default values
    if current_joint_positions is None:
        current_joint_positions = robot.get_joint_positions()
    if current_joint_velocities is None:
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

def add_return_task(controller_manager, picking_object, current_joint_positions=None, end_effector_offset=None, end_effector_orientation=None, current_observations=None, robot=None):
    # Validate picking_object
    valid_objects = ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4']
    if picking_object not in valid_objects:
        return f"Invalid picking_object: {picking_object}. Must be one of {valid_objects}"

    pour_position = current_observations[picking_object]['Pour_Position']
    return_position = current_observations[picking_object]['Return_Position']

    # Set default values
    if current_joint_positions is None:
        current_joint_positions = robot.get_joint_positions()
    if end_effector_offset is None:
        end_effector_offset = np.array([0.0, 0.0, 0.055])
    if end_effector_orientation is None:
        end_effector_orientation = [1.0, 0.0, 0.0, 0.0]  # Default quaternion orientation

    param_template = {
        "pour_position": np.array(pour_position),
        "return_position": np.array(return_position),
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
                "description": "Adds a pick-and-move task to the controller manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "picking_object": {
                            "type": "string",
                            "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                            "description": "The name of the object to pick."
                        },
                        "target_object": {
                            "type": "string",
                            "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                            "description": "The name of the target object to move to."
                        },
                        "target_position": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "The numeric target position to move the object to."
                        },
                        "current_joint_positions": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Current joint positions of the robot. Defaults to robot's current positions."
                        },
                        "end_effector_offset": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Offset for the end effector. Defaults to [0.0, 0.0, 0.06]."
                        },
                        "end_effector_orientation": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Orientation of the end effector in quaternion. Defaults to [1.0, 0.0, 0.0, 0.0]."
                        }
                    },
                    "required": ["picking_object"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_pour_task",
                "description": "Adds a pour task to the controller manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pour_speed": {
                            "type": "number",
                            "description": "Speed at which to perform the pour action."
                        },
                        "current_joint_positions": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Current joint positions of the robot. Defaults to robot's current positions."
                        },
                        "current_joint_velocities": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Current joint velocities of the robot. Defaults to robot's current velocities."
                        }
                    },
                    "required": ["pour_speed"],
                    "additionalProperties": False
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_return_task",
                "description": "Adds a return task to the controller manager.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "picking_object": {
                            "type": "string",
                            "enum": ['Bottle_Kmno4', 'Bottle_Fecl2', 'beaker_Fecl2', 'beaker_Kmno4'],
                            "description": "The name of the object to return."
                        },
                        "current_joint_positions": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Current joint positions of the robot. Defaults to robot's current positions."
                        },
                        "end_effector_offset": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Offset for the end effector. Defaults to [0.0, 0.0, 0.055]."
                        },
                        "end_effector_orientation": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Orientation of the end effector in quaternion. Defaults to [1.0, 0.0, 0.0, 0.0]."
                        }
                    },
                    "required": ["picking_object"],
                    "additionalProperties": False
                }
            }
        }
    ]
    return tools
