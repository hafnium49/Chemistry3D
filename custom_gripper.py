# custom_gripper.py

from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
import numpy as np
import torch
from typing import Callable, List

class CustomParallelGripper(ParallelGripper):
    def initialize(
        self,
        articulation_apply_action_func: Callable,
        get_joint_positions_func: Callable,
        set_joint_positions_func: Callable,
        dof_names: List,
        physics_sim_view=None,
    ) -> None:
        super().initialize(
            articulation_apply_action_func,
            get_joint_positions_func,
            set_joint_positions_func,
            dof_names,
            physics_sim_view,
        )
        current_joint_positions = get_joint_positions_func()
        if isinstance(current_joint_positions, torch.Tensor):
            # Move tensor to CPU and convert to NumPy array
            current_joint_positions_np = current_joint_positions.detach().cpu().numpy()
        else:
            # If it's already a NumPy array or list
            current_joint_positions_np = np.array(current_joint_positions)
        if self._default_state is None:
            self._default_state = np.array(
                [
                    current_joint_positions_np[self._joint_dof_indicies[0]],
                    current_joint_positions_np[self._joint_dof_indicies[1]],
                ]
            )
        self._set_joint_positions_func = set_joint_positions_func
        return
