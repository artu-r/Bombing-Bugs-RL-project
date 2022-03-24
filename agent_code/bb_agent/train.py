import pickle
from typing import List
from collections import deque

import events as e
import settings as s
import numpy as np
import agent_code.bb_agent.rl as rl


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.experience_buffer = deque(maxlen=rl.EXPERIENCE_BUFFER_EPISODES)
    self.current_episode_buffer = []
    self.tau = 1
    self.old_beta = self.beta


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    reward = rl.reward_from_events(events)
    self.current_episode_buffer.append((
        rl.extract_features(old_game_state),
        self_action,
        rl.extract_features(new_game_state),
        reward,
        ))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #compute y
    y = rl.y_for_episode(self.current_episode_buffer)
    self.experience_buffer.append((self.current_episode_buffer, y))
    
    #select batch
    current_buf_len = len(self.experience_buffer)
    mask = np.sort(np.arange(current_buf_len))
    mask = np.random.choice(mask, size=min(rl.BATCH_SIZE, current_buf_len))
    batch = []
    for i in mask:
        batch.append(self.experience_buffer[i])
    
    #compute and save new beta matrix
    self.beta = rl.update_beta(batch, self.beta)
    with open(rl.MODEL_FILE_NAME, "wb") as file:
        pickle.dump(self.beta, file)

    #measure convergence
    if self.tau % 100 == 0:
        change = np.linalg.norm(self.old_beta - self.beta)
        variance = np.max(self.beta) - np.min(self.beta)
        print(f'beta change: {change}')
        print(f'beta range: {variance}')
    
    self.current_episode_buffer = []
    self.tau += 1