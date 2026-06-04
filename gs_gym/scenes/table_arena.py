"""Table arena scene for manipulation tasks.

Provides a standard table workspace for pick-and-place and manipulation tasks.
"""

import genesis as gs
import torch

from gs_gym.scenes.flat_scene import FlatScene


class TableArena(FlatScene):
    """Table arena with configurable table size and workspace bounds.

    Extends FlatScene with a table surface suitable for tabletop manipulation.
    Used by benchmarks like Metaworld, robosuite, and LIBERO.

    Args:
        n_envs: Number of parallel environments
        args: Scene configuration arguments
        table_size: Table dimensions (width, depth, height)
        table_pos: Table position (x, y, z)
        workspace_bounds: Optional workspace bounds for object spawning
        show_viewer: Whether to show the viewer
        **kwargs: Additional arguments passed to FlatScene
    """

    def __init__(
        self,
        n_envs: int,
        args,
        table_size: tuple[float, float, float] = (0.8, 0.8, 0.05),
        table_pos: tuple[float, float, float] = (0.5, 0.0, 0.4),
        workspace_bounds: tuple[float, float, float, float, float, float] | None = None,
        show_viewer: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the table arena.

        Args:
            n_envs: Number of parallel environments
            args: Scene configuration arguments
            table_size: Table dimensions (width, depth, height)
            table_pos: Table center position (x, y, z)
            workspace_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) for object spawning
            show_viewer: Whether to show the viewer
            **kwargs: Additional arguments for FlatScene
        """
        super().__init__(
            n_envs=n_envs,
            args=args,
            show_viewer=show_viewer,
            **kwargs,
        )

        self.table_size = table_size
        self.table_pos = table_pos

        # Add table entity
        self.table = self._add_table(table_size, table_pos)

        # Set workspace bounds (default to table surface area)
        if workspace_bounds is None:
            hw, hd = table_size[0] / 2, table_size[1] / 2
            table_top = table_pos[2] + table_size[2] / 2
            self.workspace_bounds = (
                table_pos[0] - hw + 0.05,  # x_min (with margin)
                table_pos[0] + hw - 0.05,  # x_max
                table_pos[1] - hd + 0.05,  # y_min
                table_pos[1] + hd - 0.05,  # y_max
                table_top,  # z_min (table surface)
                table_top + 0.3,  # z_max (above table)
            )
        else:
            self.workspace_bounds = workspace_bounds

    def _add_table(
        self,
        size: tuple[float, float, float],
        pos: tuple[float, float, float],
    ) -> gs.Entity:
        """Add a table entity to the scene.

        Args:
            size: Table dimensions (width, depth, height)
            pos: Table center position (x, y, z)

        Returns:
            The table entity
        """
        table = self._scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=pos,
            ),
            material=gs.materials.Rigid(
                friction=1.0,
                gravity_compensation=1.0,  # Table is fixed
            ),
        )
        return table

    def sample_position_on_table(
        self,
        n_samples: int = 1,
        margin: float = 0.05,
        height_offset: float = 0.01,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample random positions on the table surface.

        Args:
            n_samples: Number of positions to sample
            margin: Margin from table edges
            height_offset: Height above table surface
            device: Torch device for the tensor

        Returns:
            (n_samples, 3) tensor of positions
        """
        if device is None:
            device = torch.device("cpu")

        hw = self.table_size[0] / 2 - margin
        hd = self.table_size[1] / 2 - margin
        table_top = self.table_pos[2] + self.table_size[2] / 2 + height_offset

        x = torch.rand(n_samples, device=device) * 2 * hw + (self.table_pos[0] - hw)
        y = torch.rand(n_samples, device=device) * 2 * hd + (self.table_pos[1] - hd)
        z = torch.full((n_samples,), table_top, device=device)

        return torch.stack([x, y, z], dim=-1)

    def sample_position_in_workspace(
        self,
        n_samples: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample random positions within the workspace bounds.

        Args:
            n_samples: Number of positions to sample
            device: Torch device for the tensor

        Returns:
            (n_samples, 3) tensor of positions
        """
        if device is None:
            device = torch.device("cpu")

        x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_bounds

        x = torch.rand(n_samples, device=device) * (x_max - x_min) + x_min
        y = torch.rand(n_samples, device=device) * (y_max - y_min) + y_min
        z = torch.rand(n_samples, device=device) * (z_max - z_min) + z_min

        return torch.stack([x, y, z], dim=-1)

    @property
    def table_surface_height(self) -> float:
        """Height of the table surface (z-coordinate)."""
        return self.table_pos[2] + self.table_size[2] / 2
