"""
API Wrapper for Backtrain Training API

This module provides a comprehensive wrapper class for interacting with the Backtrain
training API, which manages reinforcement learning training infrastructure including
training sessions, worker processes, episode management, and data exchange between
agents and environments.

Overview:
    The Wrapper class provides a high-level interface to:
    - Create and manage training sessions
    - Add and monitor worker processes
    - Track episode progression and state
    - Exchange actions, observations, and environment states
    - Handle data synchronization with retry capabilities

Architecture:
    The API follows a hierarchical structure:
    Training Session → Workers → Episodes → Steps

    Each step can have associated:
    - Action: Decision made by the agent
    - Observation: Sensor data from the environment
    - Environment: Reward, done flag, and metadata

Exception Handling:
    Two distinct patterns are used based on operation criticality:

    1. Data Exchange Operations (retryable):
       - Methods: get(), post(), get_batch(), post_batch(), get_action(),
         send_action_batch(), get_observation_batch(), send_observation(),
         get_environment_batch(), send_environment()
       - Behavior: Catch exceptions, log errors, return None/False/{}/[]
       - Rationale: Handle lagging data, network issues, and race conditions
       - Usage: Caller can implement retry logic

    2. Supervisor Operations (critical):
       - Methods: create_training_session(), add_worker(), get_workers(),
         get_episode_id(), increment_episode_id(), update_worker_status(),
         get_worker_status()
       - Behavior: Propagate exceptions to caller
       - Rationale: These operations must succeed for training to proceed
       - Usage: Caller must handle exceptions appropriately
"""

import logging

import requests

from corl.schemas.api import Endpoint
from corl.schemas.learning import Action, Environment, Observation
from corl.schemas.tracker import StepKey
from corl.utils.config import Config

logger = logging.getLogger(__name__)


class Wrapper:
    """
    API wrapper for interacting with the 'Backtrain' training API.

    This wrapper provides methods to interact with various API endpoints for managing
    reinforcement learning training sessions, workers, episodes, and data exchange
    (actions, observations, and environment states).

    Exception Handling Strategy:
    - Generic methods (get, post, get_action, get_observation, etc.): Catch and log
      exceptions, returning None/False. These methods are designed to handle lagging
      data and can be retried by the caller.
    - Supervisor methods (create_training_session, add_worker, etc.): Do not catch
      exceptions and allow them to propagate. These methods represent critical
      operations that should not fail silently.

    The wrapper can be used as a context manager to ensure proper resource cleanup:
        with Wrapper(config) as wrapper:
            wrapper.create_training_session("train1")

    Attributes:
        base_url: Base URL for the API endpoints
        session: Persistent HTTP session for making requests
        timeout: Default timeout for API requests in seconds
    """

    base_url: str
    session: requests.Session
    timeout: int

    def __init__(self, config: Config, timeout: int = 10):
        """
        Initialize the API wrapper.

        Args:
            config: Configuration object containing API_HOST and API_PORT settings.
            timeout: Default timeout for API requests in seconds (default: 10)
        """
        self.base_url = f"{config.get('API_HOST')}:{config.get('API_PORT')}/api/v1"
        self.session = requests.Session()
        self.timeout = timeout

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and close the session."""
        self.session.close()

    def close(self):
        """Close the session and release resources."""
        self.session.close()

    def get(
        self,
        endpoint: Endpoint,
        train_id: str,
        worker_id: int,
        episode_id: int,
        step: int,
    ) -> dict | None:
        """
        Perform a GET request to retrieve data from a step-based endpoint.

        This method catches and logs exceptions rather than propagating them, making it
        suitable for operations that may fail due to lagging data or race conditions.
        The caller can retry the request if None is returned.

        Args:
            endpoint: The API endpoint to query (ACTION, OBSERVATION, or ENVIRONMENT)
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode

        Returns:
            Dictionary containing the response data if successful, None otherwise

        Raises:
            Does not raise exceptions. Logs errors and returns None on failure.
        """
        url = f"{self.base_url}/{endpoint}/{train_id}/{worker_id}/{episode_id}/{step}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            logger.debug(f"GET {url} returned {result}")
            return result
        except requests.RequestException as e:
            content = (
                e.response.content.decode("utf-8", errors="replace")
                if e.response
                else str(e)
            )
            logger.debug(f"GET {url} failed with error: {content}")
        except Exception as e:
            logger.debug(f"GET {url} failed with unexpected error: {e}")
        return None

    def post(
        self,
        endpoint: Endpoint,
        train_id: str,
        worker_id: int,
        episode_id: int,
        step: int,
        data: dict,
    ) -> bool:
        """
        Perform a POST request to send data to a step-based endpoint.

        This method catches and logs exceptions rather than propagating them, making it
        suitable for operations that may fail due to lagging data or race conditions.
        The caller can retry the request if False is returned.

        Args:
            endpoint: The API endpoint to send data to (ACTION, OBSERVATION, or ENVIRONMENT)
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode
            data: Dictionary containing the data to send

        Returns:
            True if the data was successfully stored (API returns {"status": "success"}),
            False otherwise

        Raises:
            Does not raise exceptions. Logs errors and returns False on failure.
        """
        url = f"{self.base_url}/{endpoint}/{train_id}/{worker_id}/{episode_id}/{step}"

        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                logger.debug(f"POST {url} succeeded")
                return True
            else:
                logger.debug(f"POST {url} returned unexpected result: {result}")
        except requests.RequestException as e:
            content = (
                e.response.content.decode("utf-8", errors="replace")
                if e.response
                else str(e)
            )
            logger.debug(f"POST {url} failed with error: {content}")
        except Exception as e:
            logger.debug(f"POST {url} failed with unexpected error: {e}")
        return False

    @staticmethod
    def _get_batch_payload(train_id: str, step_keys: list[StepKey]) -> dict:
        """
        Build the request payload for a batch GET operation.

        Args:
            train_id: Training session identifier
            step_keys: List of StepKey objects identifying the steps to retrieve

        Returns:
            Dictionary with a ``keys`` list, each entry containing train_id,
            worker_id, episode_id, and step.
        """
        return {
            "keys": [
                {
                    "train_id": train_id,
                    "worker_id": key.worker_id,
                    "episode_id": key.episode_id,
                    "step": key.step,
                }
                for key in step_keys
            ],
        }

    @staticmethod
    def _post_batch_payload(
        train_id: str, values: list[tuple[StepKey, Action | Environment | Observation]]
    ) -> dict:
        """
        Build the request payload for a batch POST operation.

        Args:
            train_id: Training session identifier
            values: List of (StepKey, data) tuples where data is an Action,
                    Environment, or Observation to publish

        Returns:
            Dictionary with an ``items`` list, each entry containing the
            serialized key and value for one step.
        """
        return {
            "items": [
                {
                    "key": {
                        "train_id": train_id,
                        "worker_id": key.worker_id,
                        "episode_id": key.episode_id,
                        "step": key.step,
                    },
                    "value": value.model_dump(),
                }
                for key, value in values
            ],
        }

    def get_batch(
        self, endpoint: Endpoint, train_id: str, step_keys: list[StepKey]
    ) -> list[dict]:
        """
        Perform a batch GET request to retrieve data for multiple steps at once.

        Sends a POST request to the batch endpoint with the list of step keys and
        returns all available results. Missing steps are logged but not treated as
        errors. Catches and logs exceptions rather than propagating them.

        Args:
            endpoint: The API endpoint to query (ACTION, OBSERVATION, or ENVIRONMENT)
            train_id: Training session identifier
            step_keys: List of StepKey objects identifying the steps to retrieve

        Returns:
            List of result dicts (each containing ``key`` and ``value`` fields) as
            returned by the API ``results`` field. Returns an empty list on failure
            or when the ``results`` key is absent from the response.

        Raises:
            Does not raise exceptions. Logs errors and returns [] on failure.
        """
        url = f"{self.base_url}/{endpoint}/batch"
        try:
            response = self.session.post(
                url,
                json=self._get_batch_payload(train_id, step_keys),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"POST {url} returned {result}")
            if int(result.get("missing")) > 0:
                logger.debug(
                    f"POST {url} reported {result.get('missing')} missing items out of {len(step_keys)} requested."
                )
            return result["results"] if "results" in result else []
        except requests.RequestException as e:
            content = (
                e.response.content.decode("utf-8", errors="replace")
                if e.response
                else str(e)
            )
            logger.debug(f"POST {url} failed with error: {content}")
        except Exception as e:
            logger.debug(f"POST {url} failed with unexpected error: {e}")
        return []

    def post_batch(
        self,
        endpoint: Endpoint,
        train_id: str,
        values: list[tuple[StepKey, Action | Environment | Observation]],
    ) -> bool:
        """
        Perform a batch POST request to publish data for multiple steps at once.

        Catches and logs exceptions rather than propagating them. Returns False if
        the API acknowledges fewer items than were sent.

        Args:
            endpoint: The API endpoint to publish to (ACTION, OBSERVATION, or ENVIRONMENT)
            train_id: Training session identifier
            values: List of (StepKey, data) tuples where data is an Action,
                    Environment, or Observation to publish

        Returns:
            True if all items were successfully stored, False otherwise

        Raises:
            Does not raise exceptions. Logs errors and returns False on failure.
        """
        url = f"{self.base_url}/{endpoint}/batch/publish"
        try:
            response = self.session.post(
                url,
                json=self._post_batch_payload(train_id, values),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"POST {url} returned {result}")
            if int(result.get("total")) != len(values):
                logger.debug(
                    f"POST {url} returned unexpected total count: {result.get('total')} (expected {len(values)})"
                )
            else:
                return True
        except requests.RequestException as e:
            content = (
                e.response.content.decode("utf-8", errors="replace")
                if e.response
                else str(e)
            )
            logger.debug(f"POST {url} failed with error: {content}")
        except Exception as e:
            logger.debug(f"POST {url} failed with unexpected error: {e}")
        return False

    # Supervisor endpoints

    def create_training_session(self, train_id: str) -> None:
        """
        Create a new training session.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Unique identifier for the training session

        Raises:
            requests.RequestException: If the HTTP request fails
            RuntimeError: If the API returns a non-success status
        """
        url = f"{self.base_url}/supervisor/train"
        response = self.session.post(
            url, json={"train_id": train_id}, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        logger.debug(f"POST {url} returned {result}")
        if result.get("status") != "success":
            raise RuntimeError(
                f"Failed to create training session {train_id}: {result}"
            )

    def delete_training_session(self, train_id: str) -> None:
        """
        Delete an existing training session.

        Args:
            train_id: Unique identifier for the training session to delete.

        Raises:
            requests.RequestException: If the HTTP request fails.
            RuntimeError: If the API returns a non-success status.
        """
        url = f"{self.base_url}/supervisor/train/{train_id}"
        response = self.session.delete(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"DELETE {url} returned {result}")
        if result.get("status") != "success":
            raise RuntimeError(
                f"Failed to delete training session {train_id}: {result}"
            )

    def add_worker(self, train_id: str) -> int:
        """
        Add a new worker to a training session.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier

        Returns:
            The assigned worker ID

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response does not contain a worker_id
        """
        url = f"{self.base_url}/supervisor/train/{train_id}/worker"
        response = self.session.post(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"POST {url} returned {result}")
        worker_id = result.get("worker_id")
        if worker_id is None:
            raise ValueError("worker_id not in response")
        return worker_id

    def get_workers(self, train_id: str) -> list[dict[str, int | bool]]:
        """
        Get the list of all workers for a training session.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier

        Returns:
            List of worker dictionaries containing worker_id and active status.
            Returns empty list if no workers exist.

        Raises:
            requests.RequestException: If the HTTP request fails
        """
        url = f"{self.base_url}/supervisor/train/{train_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"GET {url} returned {result}")
        return result.get("workers", [])

    def get_episode_id(self, train_id: str, worker_id: int) -> int:
        """
        Get the current episode ID for a worker.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier

        Returns:
            The current episode ID for the worker

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response does not contain an episode_id
        """
        url = f"{self.base_url}/supervisor/train/{train_id}/worker/{worker_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"GET {url} returned {result}")
        episode_id = result.get("episode_id")
        if episode_id is None:
            raise ValueError(f"episode_id not in response: {result}")
        return int(episode_id)

    def increment_episode_id(self, train_id: str, worker_id: int) -> None:
        """
        Increment the episode ID for a worker.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier

        Raises:
            requests.RequestException: If the HTTP request fails
            RuntimeError: If the API returns a non-success status
        """
        url = f"{self.base_url}/supervisor/train/{train_id}/worker/{worker_id}/episode/increment"
        response = self.session.post(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"POST {url} returned {result}")
        if result.get("status") != "success":
            raise RuntimeError(
                f"Failed to increment episode ID for train {train_id}, worker {worker_id}: {result}"
            )

    def update_worker_status(self, train_id: str, worker_id: int, status: bool) -> None:
        """
        Update the active status of a worker.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier
            status: New worker status (True for active, False for inactive)

        Raises:
            requests.RequestException: If the HTTP request fails
            RuntimeError: If the API returns a non-success status
        """
        url = f"{self.base_url}/supervisor/train/{train_id}/worker/{worker_id}/status"
        response = self.session.post(
            url, json={"worker_status": status}, timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()
        logger.debug(f"POST {url} returned {result}")
        if result.get("status") != "success":
            raise RuntimeError(
                f"Failed to update worker status for train {train_id}, worker {worker_id}: {result}"
            )

    def get_worker_status(self, train_id: str, worker_id: int) -> bool:
        """
        Get the active status of a worker.

        This is a critical operation that does not catch exceptions. Failures will
        propagate to the caller as this operation should not fail silently.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier

        Returns:
            The active status of the worker (True for active, False for inactive)

        Raises:
            requests.RequestException: If the HTTP request fails
            ValueError: If the response does not contain a status field
        """
        url = f"{self.base_url}/supervisor/train/{train_id}/worker/{worker_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        logger.debug(f"GET {url} returned {result}")
        status = result.get("status")
        if status is None:
            raise ValueError(f"status not in response: {result}")
        return bool(status)

    # Action endpoints

    def get_action(
        self, train_id: str, worker_id: int, episode_id: int, step: int
    ) -> Action | None:
        """
        Retrieve an action from the API.

        This method catches exceptions internally (via the get method) and can be
        retried if None is returned, making it suitable for handling lagging data.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode

        Returns:
            Action object if found and successfully parsed, None otherwise

        Raises:
            Does not raise exceptions. Returns None on failure.
        """
        action = self.get(
            Endpoint.ACTION,
            train_id,
            worker_id,
            episode_id,
            step,
        )
        return (
            Action(action=int(action["action"]), executed=action["executed"])
            if action is not None
            else None
        )

    def send_action(
        self, train_id: str, worker_id: int, episode_id: int, step: int, action: Action
    ) -> bool:
        """
        Send an action to the API.

        This method catches exceptions internally (via the post method) and can be
        retried if False is returned, making it suitable for handling lagging data.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode
            action: Action object to send

        Returns:
            True if the action was successfully stored, False otherwise

        Raises:
            Does not raise exceptions. Returns False on failure.
        """
        stored = self.post(
            Endpoint.ACTION,
            train_id,
            worker_id,
            episode_id,
            step,
            action.model_dump(),
        )
        return stored

    def send_action_batch(
        self, train_id: str, actions: list[tuple[StepKey, Action]]
    ) -> bool:
        """
        Send a batch of actions to the API.

        This method catches exceptions internally (via the post_batch method) and can be
        retried if False is returned, making it suitable for handling lagging data.

        Args:
            train_id: Training session identifier
            actions: List of tuples containing StepKey and Action objects to send

        Returns:
            True if all actions were successfully stored, False otherwise

        Raises:
            Does not raise exceptions. Returns False on failure.
        """
        stored = self.post_batch(
            Endpoint.ACTION,
            train_id,
            actions,
        )
        return stored

    # Observation endpoints

    def get_observation_batch(
        self, train_id: str, step_keys: list[StepKey]
    ) -> list[tuple[StepKey, Observation | None]]:
        """
        Retrieve a batch of observations from the API.

        This method catches exceptions internally (via get_batch) and can be
        retried if an empty list is returned, making it suitable for handling
        lagging data.

        Args:
            train_id: Training session identifier
            step_keys: List of StepKey objects identifying the steps to retrieve

        Returns:
            List of (StepKey, Observation) tuples for all available steps.
            Returns an empty list on failure or if no data is available.

        Raises:
            Does not raise exceptions. Returns [] on failure.
        """

        observations = self.get_batch(
            Endpoint.OBSERVATION,
            train_id,
            step_keys,
        )
        return [
            (
                StepKey.model_validate(observation["key"]),
                Observation(data=observation["value"]["data"])
                if observation["value"] is not None
                else None,
            )
            for observation in observations
        ]

    def send_observation(
        self,
        train_id: str,
        worker_id: int,
        episode_id: int,
        step: int,
        observation: Observation,
    ) -> bool:
        """
        Send an observation to the API.

        This method catches exceptions internally (via the post method) and can be
        retried if False is returned, making it suitable for handling lagging data.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode
            observation: Observation object to send

        Returns:
            True if the observation was successfully stored, False otherwise

        Raises:
            Does not raise exceptions. Returns False on failure.
        """
        stored = self.post(
            Endpoint.OBSERVATION,
            train_id,
            worker_id,
            episode_id,
            step,
            observation.model_dump(),
        )
        return stored

    # Environment endpoints

    def send_environment(
        self,
        train_id: str,
        worker_id: int,
        episode_id: int,
        step: int,
        state: Environment,
    ) -> bool:
        """
        Send environment state to the API.

        This method catches exceptions internally (via the post method) and can be
        retried if False is returned, making it suitable for handling lagging data.

        Args:
            train_id: Training session identifier
            worker_id: Worker identifier within the training session
            episode_id: Episode identifier within the worker
            step: Step number within the episode
            state: Environment object containing reward, done flag, and optional data

        Returns:
            True if the environment state was successfully stored, False otherwise

        Raises:
            Does not raise exceptions. Returns False on failure.
        """
        return self.post(
            Endpoint.ENVIRONMENT,
            train_id,
            worker_id,
            episode_id,
            step,
            state.model_dump(),
        )

    def get_environment_batch(
        self, train_id: str, step_keys: list[StepKey]
    ) -> list[tuple[StepKey, Environment | None]]:
        """
        Retrieve a batch of environment states from the API.

        This method catches exceptions internally (via get_batch) and can be
        retried if an empty list is returned, making it suitable for handling
        lagging data.

        Args:
            train_id: Training session identifier
            step_keys: List of StepKey objects identifying the steps to retrieve

        Returns:
            List of (StepKey, Environment) tuples for all available steps.
            Returns an empty list on failure or if no data is available.

        Raises:
            Does not raise exceptions. Returns [] on failure.
        """

        environments = self.get_batch(
            Endpoint.ENVIRONMENT,
            train_id,
            step_keys,
        )
        return [
            (
                StepKey.model_validate(environment["key"]),
                Environment(
                    done=environment["value"]["done"],
                    reward=environment["value"]["reward"],
                    data=environment["value"].get("data", {}),
                )
                if environment["value"] is not None
                else None,
            )
            for environment in environments
        ]
