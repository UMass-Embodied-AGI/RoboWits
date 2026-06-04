"""Mutation task environments for RoboWits benchmark."""

from importlib import import_module as _import

_pkg = "gs_gym.envs.robowits.mutation"

AlignBlocksMut1Env = _import(f"{_pkg}.01_01").AlignBlocksMut1Env
AlignBlocksMut2Env = _import(f"{_pkg}.01_02").AlignBlocksMut2Env
AlignBlocksMut3Env = _import(f"{_pkg}.01_03").AlignBlocksMut3Env
AlignBlocksMut4Env = _import(f"{_pkg}.01_04").AlignBlocksMut4Env
AlignBlocksMut5Env = _import(f"{_pkg}.01_05").AlignBlocksMut5Env
AlignBlocksMut6Env = _import(f"{_pkg}.01_06").AlignBlocksMut6Env
RetrieveCubeMut1Env = _import(f"{_pkg}.02_01").RetrieveCubeMut1Env
RetrieveCubeMut2Env = _import(f"{_pkg}.02_02").RetrieveCubeMut2Env
RetrieveCubeMut6Env = _import(f"{_pkg}.02_06").RetrieveCubeMut6Env
RetrieveCubeMut7Env = _import(f"{_pkg}.02_07").RetrieveCubeMut7Env
RetrieveCubeMut10Env = _import(f"{_pkg}.02_10").RetrieveCubeMut10Env
GapRetrieveMut1Env = _import(f"{_pkg}.03_01").GapRetrieveMut1Env
GapRetrieveMut2Env = _import(f"{_pkg}.03_02").GapRetrieveMut2Env
GapRetrieveMut3Env = _import(f"{_pkg}.03_03").GapRetrieveMut3Env
GapRetrieveMut4Env = _import(f"{_pkg}.03_04").GapRetrieveMut4Env
GapRetrieveMut5Env = _import(f"{_pkg}.03_05").GapRetrieveMut5Env
PinchCardMut1Env = _import(f"{_pkg}.04_01").PinchCardMut1Env
PinchCardMut2Env = _import(f"{_pkg}.04_02").PinchCardMut2Env
PinchCardMut3Env = _import(f"{_pkg}.04_03").PinchCardMut3Env
PinchCardMut4Env = _import(f"{_pkg}.04_04").PinchCardMut4Env
PinchCardMut5Env = _import(f"{_pkg}.04_05").PinchCardMut5Env
RollUpBallMut1Env = _import(f"{_pkg}.05_01").RollUpBallMut1Env
RollUpBallMut2Env = _import(f"{_pkg}.05_02").RollUpBallMut2Env
RollUpBallMut3Env = _import(f"{_pkg}.05_03").RollUpBallMut3Env
RollUpBallMut4Env = _import(f"{_pkg}.05_04").RollUpBallMut4Env
RollUpBallMut5Env = _import(f"{_pkg}.05_05").RollUpBallMut5Env
RollUpBallMut6Env = _import(f"{_pkg}.05_06").RollUpBallMut6Env
RollUpBallMut7Env = _import(f"{_pkg}.05_07").RollUpBallMut7Env
RollUpBallMut8Env = _import(f"{_pkg}.05_08").RollUpBallMut8Env
DominosMut1Env = _import(f"{_pkg}.06_01").DominosMut1Env
DominosMut2Env = _import(f"{_pkg}.06_02").DominosMut2Env
DominosMut3Env = _import(f"{_pkg}.06_03").DominosMut3Env
DominosMut4Env = _import(f"{_pkg}.06_04").DominosMut4Env
DominosMut5Env = _import(f"{_pkg}.06_05").DominosMut5Env
DominosMut6Env = _import(f"{_pkg}.06_06").DominosMut6Env
StandPagesMut1Env = _import(f"{_pkg}.07_01").StandPagesMut1Env
RoundDoughSheetMut1Env = _import(f"{_pkg}.08_01").RoundDoughSheetMut1Env
RoundDoughSheetMut2Env = _import(f"{_pkg}.08_02").RoundDoughSheetMut2Env
RoundDoughSheetMut3Env = _import(f"{_pkg}.08_03").RoundDoughSheetMut3Env
RoundDoughSheetMut4Env = _import(f"{_pkg}.08_04").RoundDoughSheetMut4Env
RoundDoughSheetMut5Env = _import(f"{_pkg}.08_05").RoundDoughSheetMut5Env
RoundDoughSheetMut6Env = _import(f"{_pkg}.08_06").RoundDoughSheetMut6Env
HoldCupMut1Env = _import(f"{_pkg}.09_01").HoldCupMut1Env
HoldCupMut2Env = _import(f"{_pkg}.09_02").HoldCupMut2Env
HoldCupMut3Env = _import(f"{_pkg}.09_03").HoldCupMut3Env
HoldCupMut4Env = _import(f"{_pkg}.09_04").HoldCupMut4Env
HoldCupMut5Env = _import(f"{_pkg}.09_05").HoldCupMut5Env
CollectScrewsMut1Env = _import(f"{_pkg}.10_01").CollectScrewsMut1Env
CollectScrewsMut2Env = _import(f"{_pkg}.10_02").CollectScrewsMut2Env
CollectScrewsMut3Env = _import(f"{_pkg}.10_03").CollectScrewsMut3Env
CollectScrewsMut4Env = _import(f"{_pkg}.10_04").CollectScrewsMut4Env
CollectScrewsMut5Env = _import(f"{_pkg}.10_05").CollectScrewsMut5Env
PlaceTallBoxMut1Env = _import(f"{_pkg}.11_01").PlaceTallBoxMut1Env
PlaceTallBoxMut2Env = _import(f"{_pkg}.11_02").PlaceTallBoxMut2Env
PlaceTallBoxMut3Env = _import(f"{_pkg}.11_03").PlaceTallBoxMut3Env
PlaceTallBoxMut4Env = _import(f"{_pkg}.11_04").PlaceTallBoxMut4Env
PlaceTallBoxMut5Env = _import(f"{_pkg}.11_05").PlaceTallBoxMut5Env
BallIntoBottleMut1Env = _import(f"{_pkg}.12_01").BallIntoBottleMut1Env
BallIntoBottleMut2Env = _import(f"{_pkg}.12_02").BallIntoBottleMut2Env
BallIntoBottleMut3Env = _import(f"{_pkg}.12_03").BallIntoBottleMut3Env
BallIntoBottleMut4Env = _import(f"{_pkg}.12_04").BallIntoBottleMut4Env
BallIntoBottleMut5Env = _import(f"{_pkg}.12_05").BallIntoBottleMut5Env
BallIntoBottleMut6Env = _import(f"{_pkg}.12_06").BallIntoBottleMut6Env
CoverWithLidMut1Env = _import(f"{_pkg}.13_01").CoverWithLidMut1Env
CoverWithLidMut2Env = _import(f"{_pkg}.13_02").CoverWithLidMut2Env
CoverWithLidMut3Env = _import(f"{_pkg}.13_03").CoverWithLidMut3Env
CoverWithLidMut5Env = _import(f"{_pkg}.13_05").CoverWithLidMut5Env
CoverWithLidMut6Env = _import(f"{_pkg}.13_06").CoverWithLidMut6Env
StackCubesMut1Env = _import(f"{_pkg}.14_01").StackCubesMut1Env
StackCubesMut2Env = _import(f"{_pkg}.14_02").StackCubesMut2Env
StackCubesMut3Env = _import(f"{_pkg}.14_03").StackCubesMut3Env
StackCubesMut4Env = _import(f"{_pkg}.14_04").StackCubesMut4Env
StackCubesMut6Env = _import(f"{_pkg}.14_06").StackCubesMut6Env
SeparateMarblesAndSandMut1Env = _import(f"{_pkg}.15_01").SeparateMarblesAndSandMut1Env
SeparateMarblesAndSandMut3Env = _import(f"{_pkg}.15_03").SeparateMarblesAndSandMut3Env
SeparateMarblesAndSandMut4Env = _import(f"{_pkg}.15_04").SeparateMarblesAndSandMut4Env
SeparateMarblesAndSandMut5Env = _import(f"{_pkg}.15_05").SeparateMarblesAndSandMut5Env
StandBulbMut1Env = _import(f"{_pkg}.16_01").StandBulbMut1Env
StandBulbMut2Env = _import(f"{_pkg}.16_02").StandBulbMut2Env
StandBulbMut3Env = _import(f"{_pkg}.16_03").StandBulbMut3Env
StandBulbMut4Env = _import(f"{_pkg}.16_04").StandBulbMut4Env
StandBulbMut5Env = _import(f"{_pkg}.16_05").StandBulbMut5Env
StandBulbMut6Env = _import(f"{_pkg}.16_06").StandBulbMut6Env
StandBulbMut7Env = _import(f"{_pkg}.16_07").StandBulbMut7Env
BallOntoTowerMut1Env = _import(f"{_pkg}.17_01").BallOntoTowerMut1Env
BallOntoTowerMut2Env = _import(f"{_pkg}.17_02").BallOntoTowerMut2Env
BallOntoTowerMut3Env = _import(f"{_pkg}.17_03").BallOntoTowerMut3Env
BallOntoTowerMut4Env = _import(f"{_pkg}.17_04").BallOntoTowerMut4Env
BallOntoTowerMut5Env = _import(f"{_pkg}.17_05").BallOntoTowerMut5Env
BallOntoTowerMut6Env = _import(f"{_pkg}.17_06").BallOntoTowerMut6Env
BallOntoTowerMut7Env = _import(f"{_pkg}.17_07").BallOntoTowerMut7Env
BallOntoTowerMut8Env = _import(f"{_pkg}.17_08").BallOntoTowerMut8Env
CylinderThroughHoleMut1Env = _import(f"{_pkg}.18_01").CylinderThroughHoleMut1Env
CylinderThroughHoleMut2Env = _import(f"{_pkg}.18_02").CylinderThroughHoleMut2Env
CylinderThroughHoleMut3Env = _import(f"{_pkg}.18_03").CylinderThroughHoleMut3Env
CylinderThroughHoleMut4Env = _import(f"{_pkg}.18_04").CylinderThroughHoleMut4Env
CylinderThroughHoleMut5Env = _import(f"{_pkg}.18_05").CylinderThroughHoleMut5Env
CylinderThroughHoleMut6Env = _import(f"{_pkg}.18_06").CylinderThroughHoleMut6Env
StackBowlsMut1Env = _import(f"{_pkg}.19_01").StackBowlsMut1Env
StackBowlsMut2Env = _import(f"{_pkg}.19_02").StackBowlsMut2Env
StackBowlsMut3Env = _import(f"{_pkg}.19_03").StackBowlsMut3Env
StackBowlsMut5Env = _import(f"{_pkg}.19_05").StackBowlsMut5Env
StackBowlsMut6Env = _import(f"{_pkg}.19_06").StackBowlsMut6Env
StackBowlsMut7Env = _import(f"{_pkg}.19_07").StackBowlsMut7Env
BallIntoJarMut1Env = _import(f"{_pkg}.20_01").BallIntoJarMut1Env
BallIntoJarMut2Env = _import(f"{_pkg}.20_02").BallIntoJarMut2Env
BallIntoJarMut3Env = _import(f"{_pkg}.20_03").BallIntoJarMut3Env
BallIntoJarMut4Env = _import(f"{_pkg}.20_04").BallIntoJarMut4Env
BallIntoJarMut5Env = _import(f"{_pkg}.20_05").BallIntoJarMut5Env
SealColanderMut2Env = _import(f"{_pkg}.21_02").SealColanderMut2Env
SealColanderMut4Env = _import(f"{_pkg}.21_04").SealColanderMut4Env
SealColanderMut5Env = _import(f"{_pkg}.21_05").SealColanderMut5Env
StabilizeBottleMut1Env = _import(f"{_pkg}.22_01").StabilizeBottleMut1Env
StabilizeBottleMut2Env = _import(f"{_pkg}.22_02").StabilizeBottleMut2Env
StabilizeBottleMut3Env = _import(f"{_pkg}.22_03").StabilizeBottleMut3Env
StabilizeBottleMut4Env = _import(f"{_pkg}.22_04").StabilizeBottleMut4Env
StabilizeBottleMut5Env = _import(f"{_pkg}.22_05").StabilizeBottleMut5Env
PlaceBookMut1Env = _import(f"{_pkg}.23_01").PlaceBookMut1Env
PlaceBookMut2Env = _import(f"{_pkg}.23_02").PlaceBookMut2Env
PlaceBookMut3Env = _import(f"{_pkg}.23_03").PlaceBookMut3Env
PlaceBookMut4Env = _import(f"{_pkg}.23_04").PlaceBookMut4Env
PlaceBookMut5Env = _import(f"{_pkg}.23_05").PlaceBookMut5Env
RaisePlatformMut1Env = _import(f"{_pkg}.24_01").RaisePlatformMut1Env
RaisePlatformMut2Env = _import(f"{_pkg}.24_02").RaisePlatformMut2Env
RaisePlatformMut3Env = _import(f"{_pkg}.24_03").RaisePlatformMut3Env
RaisePlatformMut4Env = _import(f"{_pkg}.24_04").RaisePlatformMut4Env
WaterIntoMugMut1Env = _import(f"{_pkg}.25_01").WaterIntoMugMut1Env
WaterIntoMugMut2Env = _import(f"{_pkg}.25_02").WaterIntoMugMut2Env
WaterIntoMugMut3Env = _import(f"{_pkg}.25_03").WaterIntoMugMut3Env
WaterIntoMugMut4Env = _import(f"{_pkg}.25_04").WaterIntoMugMut4Env
WaterIntoMugMut5Env = _import(f"{_pkg}.25_05").WaterIntoMugMut5Env
WaterIntoMugMut6Env = _import(f"{_pkg}.25_06").WaterIntoMugMut6Env
WaterIntoMugMut7Env = _import(f"{_pkg}.25_07").WaterIntoMugMut7Env
AlignChopsticksMut1Env = _import(f"{_pkg}.26_01").AlignChopsticksMut1Env
AlignChopsticksMut2Env = _import(f"{_pkg}.26_02").AlignChopsticksMut2Env
AlignChopsticksMut3Env = _import(f"{_pkg}.26_03").AlignChopsticksMut3Env
AlignChopsticksMut4Env = _import(f"{_pkg}.26_04").AlignChopsticksMut4Env
AlignChopsticksMut5Env = _import(f"{_pkg}.26_05").AlignChopsticksMut5Env
RetrieveRollMut1Env = _import(f"{_pkg}.27_01").RetrieveRollMut1Env
RetrieveRollMut2Env = _import(f"{_pkg}.27_02").RetrieveRollMut2Env
RetrieveRollMut3Env = _import(f"{_pkg}.27_03").RetrieveRollMut3Env
RetrieveRollMut4Env = _import(f"{_pkg}.27_04").RetrieveRollMut4Env
RetrieveRollMut5Env = _import(f"{_pkg}.27_05").RetrieveRollMut5Env
RetrieveRollMut6Env = _import(f"{_pkg}.27_06").RetrieveRollMut6Env
RetrieveRollMut7Env = _import(f"{_pkg}.27_07").RetrieveRollMut7Env
MoveCubeMut1Env = _import(f"{_pkg}.28_01").MoveCubeMut1Env
MoveCubeMut2Env = _import(f"{_pkg}.28_02").MoveCubeMut2Env
MoveCubeMut3Env = _import(f"{_pkg}.28_03").MoveCubeMut3Env
MoveCubeMut4Env = _import(f"{_pkg}.28_04").MoveCubeMut4Env
MoveCubeMut5Env = _import(f"{_pkg}.28_05").MoveCubeMut5Env
MoveCubeMut6Env = _import(f"{_pkg}.28_06").MoveCubeMut6Env
MoveCubeMut7Env = _import(f"{_pkg}.28_07").MoveCubeMut7Env
BalanceBoardMut1Env = _import(f"{_pkg}.29_01").BalanceBoardMut1Env
BalanceBoardMut2Env = _import(f"{_pkg}.29_02").BalanceBoardMut2Env
BalanceBoardMut3Env = _import(f"{_pkg}.29_03").BalanceBoardMut3Env
BalanceBoardMut4Env = _import(f"{_pkg}.29_04").BalanceBoardMut4Env
BalanceBoardMut5Env = _import(f"{_pkg}.29_05").BalanceBoardMut5Env
BalanceBoardMut6Env = _import(f"{_pkg}.29_06").BalanceBoardMut6Env
DifferentiateCubesMut1Env = _import(f"{_pkg}.30_01").DifferentiateCubesMut1Env
DifferentiateCubesMut2Env = _import(f"{_pkg}.30_02").DifferentiateCubesMut2Env
DifferentiateCubesMut3Env = _import(f"{_pkg}.30_03").DifferentiateCubesMut3Env
DifferentiateCubesMut4Env = _import(f"{_pkg}.30_04").DifferentiateCubesMut4Env
DifferentiateCubesMut5Env = _import(f"{_pkg}.30_05").DifferentiateCubesMut5Env

__all__ = [
    "AlignBlocksMut1Env",
    "AlignBlocksMut2Env",
    "AlignBlocksMut3Env",
    "AlignBlocksMut4Env",
    "AlignBlocksMut5Env",
    "AlignBlocksMut6Env",
    "RetrieveCubeMut1Env",
    "RetrieveCubeMut2Env",
    "RetrieveCubeMut6Env",
    "RetrieveCubeMut7Env",
    "RetrieveCubeMut10Env",
    "GapRetrieveMut1Env",
    "GapRetrieveMut2Env",
    "GapRetrieveMut3Env",
    "GapRetrieveMut4Env",
    "GapRetrieveMut5Env",
    "PinchCardMut1Env",
    "PinchCardMut2Env",
    "PinchCardMut3Env",
    "PinchCardMut4Env",
    "PinchCardMut5Env",
    "RollUpBallMut1Env",
    "RollUpBallMut2Env",
    "RollUpBallMut3Env",
    "RollUpBallMut4Env",
    "RollUpBallMut5Env",
    "RollUpBallMut6Env",
    "RollUpBallMut7Env",
    "RollUpBallMut8Env",
    "DominosMut1Env",
    "DominosMut2Env",
    "DominosMut3Env",
    "DominosMut4Env",
    "DominosMut5Env",
    "DominosMut6Env",
    "StandPagesMut1Env",
    "RoundDoughSheetMut1Env",
    "RoundDoughSheetMut2Env",
    "RoundDoughSheetMut3Env",
    "RoundDoughSheetMut4Env",
    "RoundDoughSheetMut5Env",
    "RoundDoughSheetMut6Env",
    "HoldCupMut1Env",
    "HoldCupMut2Env",
    "HoldCupMut3Env",
    "HoldCupMut4Env",
    "HoldCupMut5Env",
    "CollectScrewsMut1Env",
    "CollectScrewsMut2Env",
    "CollectScrewsMut3Env",
    "CollectScrewsMut4Env",
    "CollectScrewsMut5Env",
    "PlaceTallBoxMut1Env",
    "PlaceTallBoxMut2Env",
    "PlaceTallBoxMut3Env",
    "PlaceTallBoxMut4Env",
    "PlaceTallBoxMut5Env",
    "BallIntoBottleMut1Env",
    "BallIntoBottleMut2Env",
    "BallIntoBottleMut3Env",
    "BallIntoBottleMut4Env",
    "BallIntoBottleMut5Env",
    "BallIntoBottleMut6Env",
    "CoverWithLidMut1Env",
    "CoverWithLidMut2Env",
    "CoverWithLidMut3Env",
    "CoverWithLidMut5Env",
    "CoverWithLidMut6Env",
    "StackCubesMut1Env",
    "StackCubesMut2Env",
    "StackCubesMut3Env",
    "StackCubesMut4Env",
    "StackCubesMut6Env",
    "SeparateMarblesAndSandMut1Env",
    "SeparateMarblesAndSandMut3Env",
    "SeparateMarblesAndSandMut4Env",
    "SeparateMarblesAndSandMut5Env",
    "StandBulbMut1Env",
    "StandBulbMut2Env",
    "StandBulbMut3Env",
    "StandBulbMut4Env",
    "StandBulbMut5Env",
    "StandBulbMut6Env",
    "StandBulbMut7Env",
    "BallOntoTowerMut1Env",
    "BallOntoTowerMut2Env",
    "BallOntoTowerMut3Env",
    "BallOntoTowerMut4Env",
    "BallOntoTowerMut5Env",
    "BallOntoTowerMut6Env",
    "BallOntoTowerMut7Env",
    "BallOntoTowerMut8Env",
    "CylinderThroughHoleMut1Env",
    "CylinderThroughHoleMut2Env",
    "CylinderThroughHoleMut3Env",
    "CylinderThroughHoleMut4Env",
    "CylinderThroughHoleMut5Env",
    "CylinderThroughHoleMut6Env",
    "StackBowlsMut1Env",
    "StackBowlsMut2Env",
    "StackBowlsMut3Env",
    "StackBowlsMut5Env",
    "StackBowlsMut6Env",
    "StackBowlsMut7Env",
    "BallIntoJarMut1Env",
    "BallIntoJarMut2Env",
    "BallIntoJarMut3Env",
    "BallIntoJarMut4Env",
    "BallIntoJarMut5Env",
    "SealColanderMut2Env",
    "SealColanderMut4Env",
    "SealColanderMut5Env",
    "StabilizeBottleMut1Env",
    "StabilizeBottleMut2Env",
    "StabilizeBottleMut3Env",
    "StabilizeBottleMut4Env",
    "StabilizeBottleMut5Env",
    "PlaceBookMut1Env",
    "PlaceBookMut2Env",
    "PlaceBookMut3Env",
    "PlaceBookMut4Env",
    "PlaceBookMut5Env",
    "RaisePlatformMut1Env",
    "RaisePlatformMut2Env",
    "RaisePlatformMut3Env",
    "RaisePlatformMut4Env",
    "WaterIntoMugMut1Env",
    "WaterIntoMugMut2Env",
    "WaterIntoMugMut3Env",
    "WaterIntoMugMut4Env",
    "WaterIntoMugMut5Env",
    "WaterIntoMugMut6Env",
    "WaterIntoMugMut7Env",
    "AlignChopsticksMut1Env",
    "AlignChopsticksMut2Env",
    "AlignChopsticksMut3Env",
    "AlignChopsticksMut4Env",
    "AlignChopsticksMut5Env",
    "RetrieveRollMut1Env",
    "RetrieveRollMut2Env",
    "RetrieveRollMut3Env",
    "RetrieveRollMut4Env",
    "RetrieveRollMut5Env",
    "RetrieveRollMut6Env",
    "RetrieveRollMut7Env",
    "MoveCubeMut1Env",
    "MoveCubeMut2Env",
    "MoveCubeMut3Env",
    "MoveCubeMut4Env",
    "MoveCubeMut5Env",
    "MoveCubeMut6Env",
    "MoveCubeMut7Env",
    "BalanceBoardMut1Env",
    "BalanceBoardMut2Env",
    "BalanceBoardMut3Env",
    "BalanceBoardMut4Env",
    "BalanceBoardMut5Env",
    "BalanceBoardMut6Env",
    "DifferentiateCubesMut1Env",
    "DifferentiateCubesMut2Env",
    "DifferentiateCubesMut3Env",
    "DifferentiateCubesMut4Env",
    "DifferentiateCubesMut5Env",
]
