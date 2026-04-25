import logging
from abc import ABC, abstractmethod

from controller import Supervisor

from corl.schemas.learning import Environment as EnvironmentSchema

logger = logging.getLogger(__name__)


class Environment(ABC):
    """
    Abstract base class for a Webots reinforcement learning environment.

    Subclasses must implement :meth:`evaluate_training_step`, which advances
    the simulation by one logical RL step and returns an
    :class:`~corl.schemas.learning.Environment` schema describing the resulting
    state, and :meth:`is_success`, which signals early episode termination.

    The simulation is driven by :meth:`run` in standalone mode, or by
    :class:`~corl.trainer.environment.TrainerEnvironment` during training.

    Attributes:
        supervisor: Webots ``Supervisor`` node controlling the simulation.
        timestep: Simulation timestep in milliseconds.
        timestep_index: Number of simulation timesteps elapsed in the current episode.
        max_timestep: Maximum number of simulation timesteps per episode.
    """

    supervisor: Supervisor
    timestep: int
    timestep_index: int
    max_timestep: int

    def __init__(
        self,
        supervisor: Supervisor,
        timestep: int,
        max_timestep: int,
    ):
        """
        Initialize the environment.

        Args:
            supervisor: Webots ``Supervisor`` node used to control the
                simulation.
            timestep: Simulation timestep in milliseconds passed to
                ``supervisor.step()``.
            max_timestep: Maximum number of simulation timesteps per episode.
        """
        self.supervisor = supervisor
        self.timestep = timestep
        self.timestep_index = 0
        self.max_timestep = max_timestep

    @abstractmethod
    def evaluate_training_step(self) -> EnvironmentSchema:
        """
        Advance the simulation by one logical RL step.

        Subclasses must override this method to:
            - Update or compute the new environment state.
            - Compute and return the reward for the current transition.
            - Set ``done=True`` on the returned schema when the episode
              should terminate.

        Returns:
            EnvironmentSchema: The updated environment state after the step.
        """

    @abstractmethod
    def is_success(self) -> bool:
        """
        Check if the current episode has been successfully completed.

        Returns:
            bool: True if the episode is successful, False otherwise.
        """

    def is_terminated(self) -> bool:
        """
        Check if the current episode has been terminated.

        Returns:
            bool: True if the episode is terminated, False otherwise.
        """
        return self.timestep_index >= self.max_timestep or self.is_success()

    def reset(self) -> None:
        """
        Reset the simulation and internal counters to the episode start state.

        Calls ``supervisor.simulationReset()`` and sets ``timestep_index`` back
        to 0. Should be called between episodes to ensure a clean starting state.

        Note:
            Webots applies the physics reset asynchronously. A subsequent
            ``supervisor.step(self.timestep)`` call is typically required to
            allow the reset to take effect before the next episode begins.
        """
        self.supervisor.simulationReset()
        self.timestep_index = 0
        logger.debug("Environment reset.")

    def quit(self) -> None:
        """
        Terminate the Webots simulation process with exit code 0.

        Calls ``supervisor.simulationQuit(0)``. This is a terminal action —
        the process will exit and cannot be resumed.
        """
        self.supervisor.simulationQuit(0)
        logger.debug("Environment terminated successfully.")

    def run(self) -> None:
        """
        Run the simulation loop in standalone mode (without a remote trainer).

        Advances the simulation one tick at a time, incrementing
        ``timestep_index`` each tick. The loop exits when:

        - :meth:`is_terminated` returns ``True`` (``max_timestep`` reached or
          :meth:`is_success` triggered), or
        - ``supervisor.step()`` returns ``-1`` (Webots simulation stopped).

        After the loop, :meth:`quit` is called to terminate the process.

        Note:
            For training use :class:`~corl.trainer.environment.TrainerEnvironment`
            instead, which exchanges state with the remote trainer and handles
            action selection.
        """

        while self.supervisor.step(self.timestep) != -1:
            self.timestep_index += 1

            if self.is_terminated():
                break

        logger.info(
            f"Simulation terminated in {self.supervisor.getTime()} sim-seconds."
        )

        self.quit()
