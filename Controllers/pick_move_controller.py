# pick_move_controller.py

from typing import List, Optional
import numpy as np

from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

class PickMoveController(BaseController):
    """
    A pick-and-move state machine controller.

    This controller follows a sequence of phases to pick up an object and move it to a target position.

    Phases:
    - Phase 0: Move end_effector above the object at 'end_effector_initial_height'.
    - Phase 1: Lower end_effector to grip the object.
    - Phase 2: Wait for the robot's inertia to settle.
    - Phase 3: Close the gripper to pick up the object.
    - Phase 4: Lift the object by raising the end_effector.
    - Phase 5: Move the object horizontally to the target position.
    - Phase 6: Lower the object to the target height.

    Args:
        name (str): Identifier for the controller.
        cspace_controller (BaseController): A cartesian space controller returning an ArticulationAction type.
        gripper (Gripper): A gripper controller for open/close actions.
        end_effector_initial_height (float, optional): Initial height for the end effector. Defaults to 0.32 meters if not specified.
        events_dt (list of float, optional): Time duration for each phase. Defaults to default durations divided by speed if not specified.
        speed (float, optional): Speed multiplier for phase durations. Defaults to 16.0.

    Raises:
        Exception: If 'events_dt' is not a list or numpy array.
        Exception: If 'events_dt' length is greater than 10.
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        speed: float = 16.0,
    ) -> None:
        super().__init__(name=name)
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.32 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            default_durations = [0.005, 0.005, 0.02, 0.02, 0.005, 0.005, 0.005]
            self._events_dt = [dt / speed for dt in default_durations]
        else:
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than or equal to 10")
        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False
        self._start = True
        return

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        target_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._start:
            self._start = False
            # Open the gripper at the start
            action = self._gripper.forward(action="open")
            return action
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
                print("Event:", self._event)
                print("Picking position:", picking_position)
            interpolated_xy = self._get_interpolated_xy(
                target_position[0], target_position[1], self._current_target_x, self._current_target_y
            )
            print("Interpolated XY:", interpolated_xy)
            target_height = self._get_target_hs(target_position[2])
            print("Target Height:", target_height)
            position_target = np.array(
                [
                    interpolated_xy[0] + end_effector_offset[0],
                    interpolated_xy[1] + end_effector_offset[1],
                    target_height + end_effector_offset[2],
                ]
            )
            print("Position Target:", position_target)
            if end_effector_orientation is None:
                end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )
            print("Target Joint Positions:", target_joint_positions)
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event >= 6:
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
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
        speed: float = 16.0,
    ) -> None:
        super().reset()
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        self._start = True
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than or equal to 10")
        else:
            default_durations = [0.005, 0.005, 0.02, 0.02, 0.005, 0.005, 0.005]
            self._events_dt = [dt / speed for dt in default_durations]
        return

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        self._pause = True
        return

    def resume(self) -> None:
        self._pause = False
        return
