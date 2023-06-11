import numpy as np
import math

import torch

from Model import ResNet
from util import get_encoded_state, get_valid_moves, get_value_and_terminated, get_state
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

        encoded_state = get_encoded_state(get_state(root.game))
        policy, _ = self.model(
            torch.tensor(encoded_state, device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * (self.game.size ** 2))
        valid_moves = get_valid_moves(encoded_state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = get_value_and_terminated(node.game)

            if not is_terminal:
                encoded_state = get_encoded_state(get_state(node.game))
                policy, value = self.model(torch.tensor(encoded_state, device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = get_valid_moves(encoded_state)
                policy *= valid_moves
                policy /= np.sum(policy)
                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.size ** 2)
        for child in root.children:
            action_probs[self.game.coordinate_to_scalar(self.game.recode_coordinates(child.action_taken))] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(torch.tensor(get_encoded_state(states), device=self.model.device))
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * (self.game.size ** 2))

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = get_valid_moves(get_encoded_state(states[i]))
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(spg.game, self.args)
            spg.root.visit_count = 1
            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = get_value_and_terminated(node.game)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

            if len(expandable_spGames) > 0:
                states = np.stack([get_state(spGames[mappingIdx].node.game) for mappingIdx in expandable_spGames])

                policy, value = self.model(
                    torch.tensor(get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = get_valid_moves(get_encoded_state(get_state(node.game)))
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)