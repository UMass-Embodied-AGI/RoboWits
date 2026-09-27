"""Scalar numpy surface over the rotation helpers in `math_utils.py`.

`math_utils.py` is batched torch and is the right choice inside envs and robots.
These wrappers exist only so code holding plain `(3,)` / `(4,)` numpy arrays --
the eval scripts, which talk to numpy observations and to the VLM APIs -- can
call the same functions without a tensor round-trip at every call site.

Every function here delegates; none reimplements the math. Conventions are
therefore `math_utils`' own: quaternions are `[w, x, y, z]`, euler angles are
xyz/roll-pitch-yaw in radians.

Not re-exported from `gs_gym.common.utils` -- the names would collide with the
torch versions already exported there. Import the module instead::

    from gs_gym.common.utils import np_math_utils as npm
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from gs_gym.common.utils import math_utils

FloatArray = npt.NDArray[np.float64]


def _wrap(fn: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[FloatArray], FloatArray]:
    """Lift a batched-torch rotation function to scalar float64 numpy."""

    def wrapper(array: FloatArray) -> FloatArray:
        out = fn(torch.as_tensor(np.asarray(array, dtype=np.float64)))
        return out.numpy().astype(np.float64)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = f"Scalar numpy wrapper for ``math_utils.{fn.__name__}``.\n\n{fn.__doc__ or ''}"
    return wrapper


euler_to_quat = _wrap(math_utils.euler_to_quat)
quat_to_euler = _wrap(math_utils.quat_to_euler)
quat_to_axis_angle = _wrap(math_utils.quat_to_axis_angle)
axis_angle_to_quat = _wrap(math_utils.axis_angle_to_quat)


def quat_mul(q: FloatArray, r: FloatArray) -> FloatArray:
    """Scalar numpy wrapper for ``math_utils.quat_mul`` (quaternion multiplication)."""
    out = math_utils.quat_mul(
        torch.as_tensor(np.asarray(q, dtype=np.float64)),
        torch.as_tensor(np.asarray(r, dtype=np.float64)),
    )
    return out.numpy().astype(np.float64)
