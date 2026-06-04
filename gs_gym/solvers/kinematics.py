"""Kinematics solver using Pink/Pinocchio for inverse kinematics."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np


class KinematicsSolver:
    """IK solver using Pink (Pinocchio-based) for accurate inverse kinematics.

    Supports multi-frame IK with configurable position/orientation costs.
    """

    def __init__(
        self,
        urdf_path: str,
        initial_q: np.ndarray | None = None,
        posture_cost: float | Sequence[float] = 1e-4,
        position_cost: float = 10.0,
        rotation_cost: float = 1.0,
    ) -> None:
        """Initialize the kinematics solver.

        Args:
            urdf_path: Path to robot URDF file
            initial_q: Initial joint configuration (uses neutral if None)
            posture_cost: Cost for posture task (regularization)
            position_cost: Cost for position tracking
            rotation_cost: Cost for orientation tracking
        """
        import pink
        import pinocchio as pin

        self._urdf_path = pathlib.Path(urdf_path)
        self._posture_cost = posture_cost
        self._position_cost = position_cost
        self._rotation_cost = rotation_cost

        if not self._urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")

        # Build robot model from URDF
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=self._urdf_path.as_posix(),
            package_dirs=[
                ".",
                self._urdf_path.parent.as_posix(),
                self._urdf_path.parent.parent.parent.as_posix(),
            ],
            root_joint=None,
        )

        if initial_q is None:
            initial_q = pin.neutral(self.robot.model)

        assert isinstance(initial_q, np.ndarray)
        self._initial_q = initial_q.copy()
        self._initial_configuration = pink.Configuration(self.robot.model, self.robot.data, self._initial_q)

        self.configuration = pink.Configuration(self.robot.model, self.robot.data, initial_q)

    def update_configuration(self, current_q: np.ndarray) -> None:
        """Update the current configuration for IK."""
        current_q = np.clip(
            current_q,
            self.robot.model.lowerPositionLimit,
            self.robot.model.upperPositionLimit,
        )
        self.configuration.update(current_q)

    def inverse_kinematics(
        self,
        frame_name: str | list[str],
        target_position: np.ndarray | list[np.ndarray],
        target_rotation: np.ndarray | list[np.ndarray],
        stop_threshold: float = 1e-5,
        initial_posture_cost: float = 1e-4,
        solver: str = "quadprog",
        dt: float = 0.05,
        max_solver_iters: int = 200,
    ) -> np.ndarray:
        """Solve inverse kinematics for one or more frames.

        Args:
            frame_name: Target frame name(s) in the URDF
            target_position: Target position(s) as (3,) arrays
            target_rotation: Target rotation(s) as (3, 3) rotation matrices
            stop_threshold: Error threshold for early stopping
            initial_posture_cost: Cost for staying close to initial posture
            solver: QP solver to use (None = auto-select from available)
            dt: Integration timestep
            max_solver_iters: Maximum solver iterations

        Returns:
            Joint configuration that achieves the target pose(s)
        """
        import pink
        import pinocchio as pin
        import qpsolvers
        from pink.tasks import FrameTask, PostureTask

        # Create posture task for regularization
        posture_task = PostureTask(cost=self._posture_cost)
        posture_task.set_target_from_configuration(self.configuration)

        tasks: list[Any] = [posture_task]

        # Optional: stay close to initial configuration
        if initial_posture_cost > 0:
            posture_task_initial = PostureTask(cost=initial_posture_cost)
            posture_task_initial.set_target_from_configuration(self._initial_configuration)
            tasks.append(posture_task_initial)

        # Normalize inputs to lists
        if isinstance(frame_name, str):
            frame_name = [frame_name]
            assert isinstance(target_position, np.ndarray)
            assert isinstance(target_rotation, np.ndarray)
            target_position = [target_position]
            target_rotation = [target_rotation]

        assert len(frame_name) == len(target_position) == len(target_rotation)

        # Create frame tasks for each target
        frame_tasks = []
        for fname, tpos, trot in zip(frame_name, target_position, target_rotation, strict=True):
            target_pose = pin.SE3(trot, tpos)

            frame_task = FrameTask(
                fname,
                position_cost=self._position_cost,
                orientation_cost=self._rotation_cost,
                lm_damping=1e-12,
            )
            frame_task.set_target(target_pose)
            frame_tasks.append(frame_task)

        tasks.extend(frame_tasks)

        # Auto-select solver if not specified
        if solver is None:
            for candidate in ["proxqp", "quadprog", "osqp", "clarabel"]:
                if candidate in qpsolvers.available_solvers:
                    solver = candidate
                    break
            else:
                solver = qpsolvers.available_solvers[0]

        # Validate solver availability
        if solver not in qpsolvers.available_solvers:
            available = ", ".join(qpsolvers.available_solvers)
            raise ValueError(f"'{solver}' solver not available. Available: {available}")

        # Iterative IK solving
        for _ in range(max_solver_iters):
            try:
                velocity = pink.solve_ik(self.configuration, tasks, dt, solver=solver)
            except pink.exceptions.NoSolutionFound:
                continue
            self.update_configuration(self.configuration.integrate(velocity, dt))

            # Check for early convergence
            if all(np.linalg.norm(ft.compute_error(self.configuration)) <= stop_threshold for ft in frame_tasks):
                break

        return self.configuration.q.copy()

    def forward_kinematics(self, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for a frame.

        Args:
            frame_name: Name of the frame in URDF

        Returns:
            Tuple of (position, rotation_matrix)
        """
        transform = self.configuration.get_transform(frame_name, "universe")
        return transform.translation, transform.rotation
