import numpy as np
import math
from random import choice

import torch

from REIL.HexGameRL.Model import ResNet
from engine import hex_engine
from copy import deepcopy


class Node:
    def __init__(self, game, args, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            # pick the child for which the opponent is least likely to win - value scaled from [-1, 1] to [0, 1]
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * child.prior * math.sqrt(self.visit_count / (child.visit_count + 1))

    def expand(self, policy):
        for action_idx, prob in enumerate(policy):
            if prob > 0:
                child_game = deepcopy(self.game)
                # all nodes play as if they were player 1 - therefore:
                child_game.board = child_game.recode_black_as_white()
                child_game.player = -1
                # moove has to be after recoding as it has the evaluation function in it
                child_game.moove(self.game.recode_coordinates(self.game.scalar_to_coordinates(action_idx)))
                child = Node(child_game, self.args, self, self.game.recode_coordinates(self.game.scalar_to_coordinates(action_idx)), prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self):
        root = Node(self.game, self.args)
        root.visit_count = 1

        state = np.array(root.game.board)
        encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)
        policy, _ = self.model(torch.tensor(encoded_state, device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * (self.game.size ** 2))
        valid_move_fields = encoded_state[1].reshape(-1)
        policy *= valid_move_fields
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = node.game.winner, node.game.winner != 0 or len(node.game.get_action_space()) == 0

            if not is_terminal:
                state = np.array(node.game.board)
                encoded_state = np.stack((state == 1, state == 0, state == -1)).astype(np.float32)
                policy, value = self.model(torch.tensor(encoded_state, device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_move_fields = encoded_state[1].reshape(-1)
                policy *= valid_move_fields
                policy /= np.sum(policy)
                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(len(self.game.get_action_space()))
        for child in root.children:
            action_probs[self.game.get_action_space().index(self.game.recode_coordinates(child.action_taken))] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


if __name__ == "__main__":
    args = {
        'C': 2,
        'num_searches': 1000
    }

    game = hex_engine.hexPosition(size=3)
    model = ResNet(game, 4, 64)
    model.eval()
    mcts = MCTS(game, args, model)

    while True:
        game.print()

        if game.player == 1:
            valid_moves = game.get_action_space()
            print(valid_moves)
            action = tuple(map(int, input(f'action:').split(',')))

            if action not in valid_moves:
                print("not valid")
                continue

        else:
            game.board = game.recode_black_as_white()
            game.player = 1
            mcts_probs = mcts.search()
            action = game.recode_coordinates(game.get_action_space()[np.argmax(mcts_probs)])
            game.board = game.recode_black_as_white()
            game.player = -1

        game.moove(action)
        if game.winner != 0:
            print(game.winner)
            break

