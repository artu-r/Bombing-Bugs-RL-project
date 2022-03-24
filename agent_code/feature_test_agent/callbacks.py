import agent_code.bb_agent.rl as rl

"""
A user controlled agent that prints the feature vector with each turn.
Use to test feature extraction functions
"""

def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    print(rl.extract_features(game_state))

    return game_state['user_input']
