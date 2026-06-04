"""RoboWits environments for Genesis simulator.

Manipulation benchmark environments from the RoboWits suite.
"""

from importlib import import_module as _import

from gs_gym.envs.robowits import mutation as _mutation  # registers all mutation tasks  # noqa: F401

# Placement utilities
from gs_gym.envs.robowits.placement import (
    check_collinearity,
    check_collision_2d,
    check_minimum_spread,
    check_objects_on_table,
    compute_2d_convex_hull,
    get_object_bounds_2d,
    get_object_reachable_area,
    get_oriented_hull_2d,
    parse_reachable_bounds,
    place_group_together,
    place_objects_randomly,
    place_objects_randomly_batched,
    place_single_object,
)

# Base class and types
from gs_gym.envs.robowits.robowits import (
    PlacementGroup,
    PlacementGroups,
    RoboWitsEnv,
)

# Geometry utilities
from gs_gym.envs.robowits.utils import (
    ensure_ccw,
    polygon_area,
    polygon_signed_area,
    sutherland_hodgman_clip,
)

# Task environments (01-30)
# Module names start with digits, so we use importlib.
_pkg = "gs_gym.envs.robowits"

AlignBlocksEnv = _import(f"{_pkg}.01_align_blocks").AlignBlocksEnv
RetrieveCubeEnv = _import(f"{_pkg}.02_retrieve_cube").RetrieveCubeEnv
GapRetrieveEnv = _import(f"{_pkg}.03_gap_retrieve").GapRetrieveEnv
PinchCardEnv = _import(f"{_pkg}.04_pinch_card").PinchCardEnv
RollUpBallEnv = _import(f"{_pkg}.05_roll_up_ball").RollUpBallEnv
DominosEnv = _import(f"{_pkg}.06_dominos").DominosEnv
StandPagesEnv = _import(f"{_pkg}.07_stand_pages").StandPagesEnv
RoundDoughSheetEnv = _import(f"{_pkg}.08_round_dough_sheet").RoundDoughSheetEnv
HoldCupEnv = _import(f"{_pkg}.09_hold_cup").HoldCupEnv
CollectScrewsEnv = _import(f"{_pkg}.10_collect_screws").CollectScrewsEnv
PlaceTallBoxEnv = _import(f"{_pkg}.11_place_tall_box").PlaceTallBoxEnv
BallIntoBottleEnv = _import(f"{_pkg}.12_ball_into_bottle").BallIntoBottleEnv
CoverWithLidEnv = _import(f"{_pkg}.13_cover_with_lid").CoverWithLidEnv
StackCubesEnv = _import(f"{_pkg}.14_stack_cubes").StackCubesEnv
SeparateMarblesAndSandEnv = _import(f"{_pkg}.15_separate_marbles_and_sand").SeparateMarblesAndSandEnv
StandBulbEnv = _import(f"{_pkg}.16_stand_bulb").StandBulbEnv
BallOntoTowerEnv = _import(f"{_pkg}.17_ball_onto_tower").BallOntoTowerEnv
CylinderThroughHoleEnv = _import(f"{_pkg}.18_cylinder_through_hole").CylinderThroughHoleEnv
StackBowlsEnv = _import(f"{_pkg}.19_stack_bowls").StackBowlsEnv
BallIntoJarEnv = _import(f"{_pkg}.20_ball_into_jar").BallIntoJarEnv
SealColanderEnv = _import(f"{_pkg}.21_seal_colander").SealColanderEnv
StabilizeBottleEnv = _import(f"{_pkg}.22_stabilize_bottle").StabilizeBottleEnv
PlaceBookEnv = _import(f"{_pkg}.23_place_book").PlaceBookEnv
RaisePlatformEnv = _import(f"{_pkg}.24_raise_platform").RaisePlatformEnv
WaterIntoMugEnv = _import(f"{_pkg}.25_water_into_mug").WaterIntoMugEnv
AlignChopsticksEnv = _import(f"{_pkg}.26_align_chopsticks").AlignChopsticksEnv
RetrieveRollEnv = _import(f"{_pkg}.27_retrieve_roll").RetrieveRollEnv
MoveCubeEnv = _import(f"{_pkg}.28_move_cube").MoveCubeEnv
BalanceBoardEnv = _import(f"{_pkg}.29_balance_board").BalanceBoardEnv
DifferentiateCubesEnv = _import(f"{_pkg}.30_differentiate_cubes").DifferentiateCubesEnv

__all__ = [
    # Base class and types
    "RoboWitsEnv",
    "PlacementGroup",
    "PlacementGroups",
    # Task environments
    "AlignBlocksEnv",
    "RetrieveCubeEnv",
    "GapRetrieveEnv",
    "DominosEnv",
    "StackCubesEnv",
    "PinchCardEnv",
    "RollUpBallEnv",
    "StandPagesEnv",
    "RoundDoughSheetEnv",
    "HoldCupEnv",
    "CollectScrewsEnv",
    "PlaceTallBoxEnv",
    "BallIntoBottleEnv",
    "CoverWithLidEnv",
    "SeparateMarblesAndSandEnv",
    "StandBulbEnv",
    "BallOntoTowerEnv",
    "CylinderThroughHoleEnv",
    "StackBowlsEnv",
    "BallIntoJarEnv",
    "SealColanderEnv",
    "StabilizeBottleEnv",
    "PlaceBookEnv",
    "RaisePlatformEnv",
    "WaterIntoMugEnv",
    "AlignChopsticksEnv",
    "RetrieveRollEnv",
    "MoveCubeEnv",
    "BalanceBoardEnv",
    "DifferentiateCubesEnv",
    # Placement utilities
    "check_collinearity",
    "check_collision_2d",
    "check_minimum_spread",
    "check_objects_on_table",
    "compute_2d_convex_hull",
    "get_object_bounds_2d",
    "get_object_reachable_area",
    "get_oriented_hull_2d",
    "parse_reachable_bounds",
    "place_group_together",
    "place_objects_randomly",
    "place_objects_randomly_batched",
    "place_single_object",
    # Geometry utilities
    "ensure_ccw",
    "polygon_area",
    "polygon_signed_area",
    "sutherland_hodgman_clip",
]
