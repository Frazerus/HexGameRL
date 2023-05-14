import numpy as np
import math
from random import choice
from engine import hex_engine
from copy import deepcopy


class Node:
    def __init__(self, game, args, parent=None, action_taken=None):
        self.game = game
        self.args = args
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = self.game.get_action_space()

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0

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
        # pick the child for which the opponent is least likely to win - value scaled from [-1, 1] to [0, 1]
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = choice(self.expandable_moves)
        self.expandable_moves.remove(action)
        child_game = deepcopy(self.game)
        # all nodes play as if they were player 1 - therefore:
        child_game.board = child_game.recode_black_as_white()
        child_game.player = -1
        # moove has to be after recoding as it has the evaluation function in it
        child_game.moove(self.game.recode_coordinates(action))
        child = Node(child_game, self.args, self, self.game.recode_coordinates(action))
        self.children.append(child)
        return child

    def simulate(self):
        simulation = deepcopy(self.game)
        while simulation.winner == 0 and len(simulation.get_action_space()) != 0:
            simulation.moove(choice(simulation.get_action_space()))
        return simulation.winner

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self):
        root = Node(self.game, self.args)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = node.game.winner, node.game.winner != 0 or len(node.game.get_action_space()) == 0

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(len(self.game.get_action_space()))
        for child in root.children:
            action_probs[self.game.get_action_space().index(self.game.recode_coordinates(child.action_taken))] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


args = {
    'C': 1.41,
    'num_searches': 1000
}
game = hex_engine.hexPosition()
mcts = MCTS(game, args)

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

