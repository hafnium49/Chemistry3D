# pick_move_controller.py

from typing import List, Optional
import numpy as np

import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction


class PickMoveController(manipulators_controllers.PickPlaceController):
    """
    A pick-and-move controller for the Franka robot.

    This controller extends the PickPlaceController and adjusts it for a pick-and-move task.

    Phases:
    - Phase 0: Move end effector above the object at 'end_effector_initial_height'.
    - Phase 1: Lower end effector to grip the object.
    - Phase 2: Wait for the robot's inertia to settle.
    - Phase 3: Close the gripper to pick up the object.
    - Phase 4: Lift the object by raising the end effector.
    - Phase 5: Move the object horizontally to the target position.
    - Phase 6: Lower the object to the target height.

    Args:
        name (str): Identifier for the controller.
        gripper (ParallelGripper): A gripper controller for open/close actions.
        robot_articulation (Articulation): The robot articulation.
        end_effector_initial_height (Optional[float], optional): Initial height for the end effector. Defaults to None.
        events_dt (Optional[List[float]], optional): Time duration for each phase. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        gripper: ParallelGripper,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        speed: float = 1.0
    ) -> None:
        if events_dt is None:
            # Adjusted durations for the 7 phases
            events_dt = [0.008, 0.005, 1.0, 0.1, 0.05, 0.05, 0.05] / speed
        super().__init__(
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )

    def forward(
        self,
        picking_position: np.ndarray,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Execute one step of the controller.

        Args:
            picking_position (np.ndarray): Position of the object to be picked.
            target_position (np.ndarray): Position to move the object to.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (np.ndarray, optional): Offset of the end effector target. Defaults to None.
            end_effector_orientation (np.ndarray, optional): Orientation of the end effector. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController.
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
        if self._event == 2:
            # Wait phase
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            # Close gripper
            target_joint_positions = self._gripper.forward(action="close")
        else:
            if self._event in [0, 1]:
                self._current_target_x = picking_position[0]
                self._current_target_y = picking_position[1]
                self._h0 = picking_position[2]
            interpolated_xy = self._get_interpolated_xy(
                target_position[0], target_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(target_position[2])
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event == 6:
            return 1.0
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1  # Initial height above the object
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)  # Lowering to the object's height
        elif self._event == 3:
            h = self._h0  # At object's height while gripping
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)  # Lifting the object
        elif self._event == 5:
            h = self._h1  # Moving horizontally at initial height
        elif self._event == 6:
            h = target_height  # Lowering to the target height
        else:
            raise ValueError()
        return h

    def reset(
        self,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        """
        Reset the state machine to start from the first phase.

        Args:
            end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to None.
            events_dt (list of float, optional): Time duration for each phase. Defaults to None.
        """
        super().reset(end_effector_initial_height=end_effector_initial_height, events_dt=events_dt)

    def is_done(self) -> bool:
        """
        Check if the state machine has reached the last phase.

        Returns:
            bool: True if the last phase is reached, False otherwise.
        """
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        """Pause the state machine's time and phase."""
        self._pause = True

    def resume(self) -> None:
        """Resume the state machine's time and phase."""
        self._pause = False
