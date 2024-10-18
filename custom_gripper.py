# custom_gripper.py

from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
import numpy as np
import torch
from typing import Callable, List

class CustomParallelGripper(ParallelGripper):
    def __init__(
        self,
        end_effector_prim_path: str,
        joint_prim_names: List[str],
        joint_opened_positions: np.ndarray,
        joint_closed_positions: np.ndarray,
        action_deltas: np.ndarray = None,
    ) -> None:
        super().__init__(
            end_effector_prim_path=end_effector_prim_path,
            joint_prim_names=joint_prim_names,
            joint_opened_positions=joint_opened_positions,
            joint_closed_positions=joint_closed_positions,
            action_deltas=action_deltas,
        )
        # Convert positions to torch tensors
        self._joint_opened_positions = torch.tensor(self._joint_opened_positions, dtype=torch.float32)
        self._joint_closed_positions = torch.tensor(self._joint_closed_positions, dtype=torch.float32)
        if self._action_deltas is not None:
            self._action_deltas = torch.tensor(self._action_deltas, dtype=torch.float32)
        return

    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view=None,
    ) -> None:
        super(ParallelGripper, self).initialize(physics_sim_view=physics_sim_view)
        self._get_joint_positions_func = get_joint_positions_func
        self._articulation_num_dofs = len(dof_names)
        self._joint_dof_indicies = [None, None]
        for index in range(len(dof_names)):
            if self._joint_prim_names[0] == dof_names[index]:
                self._joint_dof_indicies[0] = index
            elif self._joint_prim_names[1] == dof_names[index]:
                self._joint_dof_indicies[1] = index
        # Make sure that all gripper dof names were resolved
        if self._joint_dof_indicies[0] is None or self._joint_dof_indicies[1] is None:
            raise Exception("Not all gripper dof names were resolved to dof handles and dof indices.")

        current_joint_positions = get_joint_positions_func()
        self._device = current_joint_positions.device  # Store the device

        # Convert joint indices to a tensor
        self._joint_dof_indices_tensor = torch.tensor(self._joint_dof_indicies, dtype=torch.long, device=self._device)

        # Move positions to the correct device
        self._joint_opened_positions = self._joint_opened_positions.to(self._device)
        self._joint_closed_positions = self._joint_closed_positions.to(self._device)
        if self._action_deltas is not None:
            self._action_deltas = self._action_deltas.to(self._device)

        if self._default_state is None:
            indices = self._joint_dof_indicies
            positions = current_joint_positions[indices]
            self._default_state = positions  # Keep as torch tensor

        self._articulation_apply_action_func = articulation_apply_action_func
        self._set_joint_positions_func = set_joint_positions_func
        return

    def post_reset(self):
        positions = self._default_state
        self._set_joint_positions_func(
            positions=positions,
            joint_indices=self._joint_dof_indices_tensor,
        )
        return
