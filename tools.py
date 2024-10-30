# tools.py

def add_pickmove_task(picking_position, target_position, current_joint_positions, end_effector_offset, end_effector_orientation):
    param_template = {
        "picking_position": np.array(picking_position),
        "target_position": np.array(target_position),
        "current_joint_positions": np.array(current_joint_positions),
        "end_effector_offset": np.array(end_effector_offset),
        "end_effector_orientation": np.array(end_effector_orientation)
    }
    controller_manager.add_task('pickmove_controller', param_template)
    return "PickMove task added successfully."

def add_pour_task(franka_art_controller, current_joint_positions, current_joint_velocities, pour_speed):
    param_template = {
        "franka_art_controller": franka_art_controller,
        "current_joint_positions": np.array(current_joint_positions),
        "current_joint_velocities": np.array(current_joint_velocities),
        "pour_speed": pour_speed
    }
    controller_manager.add_task('pour_controller', param_template)
    return "Pour task added successfully."

def add_return_task(pour_position, return_position, current_joint_positions, end_effector_offset, end_effector_orientation):
    param_template = {
        "pour_position": np.array(pour_position),
        "return_position": np.array(return_position),
        "current_joint_positions": np.array(current_joint_positions),
        "end_effector_offset": np.array(end_effector_offset),
        "end_effector_orientation": np.array(end_effector_orientation)
    }
    controller_manager.add_task('return_controller', param_template)
    return "Return task added successfully."

def get_function_schemas():
    function_schemas = [
        {
            "name": "add_pickmove_task",
            "description": "Adds a pick-and-move task to the controller manager.",
            "parameters": {
                "type": "object",
                "properties": {
                    "picking_position": {"type": "array", "items": {"type": "number"}, "description": "The position to pick the object from."},
                    "target_position": {"type": "array", "items": {"type": "number"}, "description": "The target position to move the object to."},
                    "current_joint_positions": {"type": "array", "items": {"type": "number"}, "description": "Current joint positions of the robot."},
                    "end_effector_offset": {"type": "array", "items": {"type": "number"}, "description": "Offset for the end effector."},
                    "end_effector_orientation": {"type": "array", "items": {"type": "number"}, "description": "Orientation of the end effector in quaternion."}
                },
                "required": ["picking_position", "target_position", "current_joint_positions", "end_effector_offset", "end_effector_orientation"]
            }
        },
        {
            "name": "add_pour_task",
            "description": "Adds a pour task to the controller manager.",
            "parameters": {
                "type": "object",
                "properties": {
                    "franka_art_controller": {"type": "string", "description": "Franka articulation controller."},
                    "current_joint_positions": {"type": "array", "items": {"type": "number"}, "description": "Current joint positions of the robot."},
                    "current_joint_velocities": {"type": "array", "items": {"type": "number"}, "description": "Current joint velocities of the robot."},
                    "pour_speed": {"type": "number", "description": "Speed at which to perform the pour action."}
                },
                "required": ["franka_art_controller", "current_joint_positions", "current_joint_velocities", "pour_speed"]
            }
        },
        {
            "name": "add_return_task",
            "description": "Adds a return task to the controller manager.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pour_position": {"type": "array", "items": {"type": "number"}, "description": "Position where the pour was performed."},
                    "return_position": {"type": "array", "items": {"type": "number"}, "description": "Position to return the object to."},
                    "current_joint_positions": {"type": "array", "items": {"type": "number"}, "description": "Current joint positions of the robot."},
                    "end_effector_offset": {"type": "array", "items": {"type": "number"}, "description": "Offset for the end effector."},
                    "end_effector_orientation": {"type": "array", "items": {"type": "number"}, "description": "Orientation of the end effector in quaternion."}
                },
                "required": ["pour_position", "return_position", "current_joint_positions", "end_effector_offset", "end_effector_orientation"]
            }
        }
    ]
    return function_schemas
