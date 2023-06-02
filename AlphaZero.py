import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import random
from random import shuffle

from REIL.HexGameRL.Eval import Eval
from REIL.HexGameRL.MCTS import MCTSParallel
from REIL.HexGameRL.Model import ResNet
from REIL.HexGameRL.engine import hex_engine
from REIL.HexGameRL.util import get_encoded_state, get_state, get_value_and_terminated


class SelfPlayGame:
    def __init__(self, game):
        self.game = deepcopy(game)
        self.actual_player = 1
        self.memory = []
        self.root = None
        self.node = None

    def to_neutral_state(self):
        if self.game.player == -1:
            self.game.board = self.game.recode_black_as_white()
            self.game.player = 1
            self.actual_player = -1

    def to_original_state(self):
        if self.actual_player == -1:
            self.game.board = self.game.recode_black_as_white()
            self.game.player = 1
            self.actual_player = 1


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        self.game.reset()
        return_memory = []
        spGames = [SelfPlayGame(self.game) for _ in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            for spg in spGames:
                spg.to_neutral_state()
            states = np.stack([get_state(spg.game) for spg in spGames])
            self.mcts.search(states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                action_probs = np.zeros(self.game.size ** 2)
                for child in spg.root.children:
                    action_probs[self.game.coordinate_to_scalar(
                        self.game.recode_coordinates(child.action_taken))] = child.visit_count
                action_probs /= np.sum(action_probs)
                spg.memory.append((get_state(spg.root.game), action_probs, spg.actual_player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= sum(temperature_action_probs)
                action = spg.game.scalar_to_coordinates(
                    np.random.choice(self.game.size ** 2, p=temperature_action_probs))

                spg.game.moove(action)

                value, is_terminal = get_value_and_terminated(spg.game)

                if is_terminal:
                    for hist_game_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == spg.actual_player else -value
                        return_memory.append((
                            get_encoded_state(hist_game_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            for spg in spGames:
                spg.to_original_state()

        return return_memory

    def train(self, memory):
        shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory), batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        start_time = time.perf_counter()
        print('STARTED AT:', str(start_time))
        ev = Eval(self.args, size=self.args['game_size'])
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfplays in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                print(f'ITERATION {iteration} - SELFPLAY {selfplays}')
                memory += self.selfPlay()

            self.model.train()
            print(f'TRAINING - MEMORY LEN: {len(memory)}')
            for epoch in range(self.args['num_epochs']):
                print(f'ITERATION {iteration} - EPOCH {epoch}')
                self.train(memory)

            self.model.eval()
            ev.load_models()

            memory = []

            print(f'ITERATION {iteration} - EVAL VS RANDOM')
            win_loss = [ev.model_vs_random(self.model, self.args['num_eval_games'])]
            print(str(win_loss[0][0] / self.args['num_eval_games'] * 100))

            if len(ev.models) != 0:
                for idx, m in enumerate(ev.models):
                    print(f'ITERATION {iteration} - VS MODEL {idx + 1}:')
                    if self.args['reinforce']:
                        wr, reinf_memory = ev.model_vs_model_mem(self.model, m, min(1, self.args['num_selfPlay_iterations'] // len(ev.models)))
                        win_loss.append(wr)
                        memory += reinf_memory
                    else:
                        win_loss.append(ev.model_vs_model(self.model, m, min(1, self.args['num_selfPlay_iterations'] // len(ev.models))))

            with open(os.path.join(os.getcwd(), self.args['result_output_file']), 'a+') as f:
                for wl in win_loss:
                    f.write(str(wl[0] / (wl[0] + wl[1]) * 100) + ',')
                for _ in range(len(win_loss), self.args['num_iterations'] + 2):
                    f.write(',')
                f.write('\n')
            print([str(wl[0] / (wl[0] + wl[1]) * 100) for wl in win_loss])

            torch.save(self.model.state_dict(), os.path.join(os.getcwd(), self.args['model_output_folder'], f"model_{iteration + 1}.pt"))
            torch.save(self.optimizer.state_dict(),
                       os.path.join(os.getcwd(), self.args['model_output_folder'], f"optimizer_{iteration + 1}.pt"))

            if self.args['reinforce']:
                memory = [entry for entry in memory if entry[2] == -1]
                if len(memory) != 0:
                    self.model.train()
                    for epoch in range(self.args['num_reinforce_epochs']):
                        print(f'ITERATION {iteration} - REINFORCE EPOCH {epoch}')
                        self.train(memory)
            elapsed = time.perf_counter() - start_time
            if elapsed > self.args['time_limit_s'] or iteration == (self.args['num_iterations'] - 1):
                print('ENDED AT:', str(start_time + elapsed))
                print('ELAPSED:', str(elapsed))
                print(str(iteration + 1), 'ITERATIONS')
                exit(0)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    args = {
        'C': 1.4,
        'num_searches': 150,
        'num_iterations': 100,
        'num_selfPlay_iterations': 100,
        'num_parallel_games': 20,
        'num_epochs': 4,
        'batch_size': 32,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'num_eval_games': 1000,
        'game_size': 3,
        'reinforce': True,
        'num_reinforce_epochs': 4,
        'model_output_folder': 'models',
        'result_output_file': 'results.csv',
        'time_limit_s': 36000
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = hex_engine.hexPosition(size=args['game_size'])
    model = ResNet(game, 9, 128, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    ev = Eval(args, size=args['game_size'])
    print('INIT VS RANDOM ASSESSMENT')
    baseline = ev.model_vs_random(model, args['num_eval_games'])[0] / args['num_eval_games'] * 100
    print(str(baseline))
    with open(os.path.join(os.getcwd(), args['result_output_file']), 'a+') as f:
        f.write(str(baseline) + ',')
        for _ in range(args['num_iterations'] + 1):
            f.write(',')
        f.write('\n')
    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()
