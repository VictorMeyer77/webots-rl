from brain.controller.epuck import EpuckTurner


class EpuckTurnerNaive(EpuckTurner):

    @staticmethod
    def policy(observation: dict) -> int:

        right_obstacle = (
            observation["distance_sensors"][0] > 80.0
            or observation["distance_sensors"][1] > 80.0
            or observation["distance_sensors"][2] > 80.0
        )
        left_obstacle = (
            observation["distance_sensors"][5] > 80.0
            or observation["distance_sensors"][6] > 80.0
            or observation["distance_sensors"][7] > 80.0
        )

        if left_obstacle:
            return 2
        elif right_obstacle:
            return 1
        else:
            return 0
