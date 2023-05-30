import numpy as np


def get_value_and_terminated(game):
    return game.winner, game.winner != 0 or len(game.get_action_space()) == 0


def get_encoded_state(state):
    encoded_state = np.stack(
        (state == -1, state == 0, state == 1)
    ).astype(np.float32)

    if len(state.shape) == 3:
        encoded_state = np.swapaxes(encoded_state, 0, 1)

    return encoded_state


def get_state(game):
    return np.array(game.board)


def get_valid_moves(state):
    return state[1].reshape(-1)

