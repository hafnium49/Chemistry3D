# custom_franka.py

from omni.isaac.franka.franka import Franka
from omni.isaac.examples.user_examples.custom_gripper import CustomParallelGripper
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.controllers.articulation_controller import ArticulationController
from omni.isaac.core.utils.types import ArticulationAction, ArticulationActions
from omni.isaac.core.articulations import Articulation
from omni.isaac.nucleus import get_assets_root_path
import carb
import numpy as np
import torch
from typing import List, Optional

class CustomArticulationController(ArticulationController):
    def __init__(self):
        super().__init__()
    
    def apply_action(self, control_actions: ArticulationAction) -> None:
        applied_actions = self.get_applied_action()
        joint_positions = control_actions.joint_positions
        joint_indices = control_actions.joint_indices
        if joint_indices is None:
            joint_indices = self._articulation_view._backend_utils.resolve_indices(
                joint_indices, applied_actions.joint_positions.shape[0], self._articulation_view._device
            )
        else:
            joint_indices = control_actions.joint_indices

        if joint_positions is not None:
            joint_positions = self._articulation_view._backend_utils.convert(
                joint_positions, device=self._articulation_view._device
            )
            joint_positions = self._articulation_view._backend_utils.expand_dims(joint_positions, 0)
            for i in range(control_actions.get_length()):
                if joint_positions[0][i] is None or torch.isnan(joint_positions[0][i]):
                    joint_positions[0][i] = applied_actions.joint_positions[joint_indices[i]]
        joint_velocities = control_actions.joint_velocities
        if joint_velocities is not None:
            joint_velocities = self._articulation_view._backend_utils.convert(
                joint_velocities, device=self._articulation_view._device
            )
            joint_velocities = self._articulation_view._backend_utils.expand_dims(joint_velocities, 0)
            for i in range(control_actions.get_length()):
                if joint_velocities[0][i] is None or torch.isnan(joint_velocities[0][i]):
                    joint_velocities[0][i] = applied_actions.joint_velocities[joint_indices[i]]
        joint_efforts = control_actions.joint_efforts
        if joint_efforts is not None:
            joint_efforts = self._articulation_view._backend_utils.convert(
                joint_efforts, device=self._articulation_view._device
            )
            joint_efforts = self._articulation_view._backend_utils.expand_dims(joint_efforts, 0)
            for i in range(control_actions.get_length()):
                if joint_efforts[0][i] is None or torch.isnan(joint_efforts[0][i]):
                    joint_efforts[0][i] = 0
        self._articulation_view.apply_action(
            ArticulationActions(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_efforts=joint_efforts,
                joint_indices=control_actions.joint_indices,
            )
        )
        return

class CustomFranka(Franka):
    def __init__(
        self,
        prim_path: str,
        name: str = "franka_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
    ) -> None:
        # Correct use of super()
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None
        )
        if gripper_dof_names is None:
            gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"]
        if gripper_open_position is None:
            gripper_open_position = np.array([0.05, 0.05]) / get_stage_units()
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])
        if deltas is None:
            deltas = np.array([0.05, 0.05]) / get_stage_units()

        self._gripper = CustomParallelGripper(
            end_effector_prim_path=prim_path + "/panda_rightfinger",
            joint_prim_names=gripper_dof_names,
            joint_opened_positions=gripper_open_position,
            joint_closed_positions=gripper_closed_position,
            action_deltas=deltas,
        )
        return

    def initialize(self, physics_sim_view=None) -> None:
        # Correct use of super()
        super().initialize(physics_sim_view)
        # Initialize the custom articulation controller
        self._articulation_controller = CustomArticulationController()
        self._articulation_controller.initialize(articulation_view=self)
        # Initialize the end effector and the custom gripper
        self._end_effector = RigidPrim(prim_path=self.prim_path + "/panda_rightfinger", name=self.name + "_end_effector")
        self._end_effector.initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return
