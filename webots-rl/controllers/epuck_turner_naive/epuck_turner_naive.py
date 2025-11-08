import sys

sys.path.append("../../libraries")

from brain.controller.epuck.naive.epuck_turner_naive import EpuckTurnerNaive
from controller import Robot

TIME_STEP = 64
MAX_SPEED = 6.28

robot = Robot()

epuck = EpuckTurnerNaive(robot, TIME_STEP, MAX_SPEED)
epuck.run()
