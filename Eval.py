from os import listdir
from os.path import isfile
from random import choice

import numpy as np
import torch

from REIL.HexGameRL.Model import ResNet
from REIL.HexGameRL.engine import hex_engine
import os

from REIL.HexGameRL.util import get_state, get_encoded_state, get_valid_moves


class Eval:
    def __init__(self, args, size=7):
        self.game = hex_engine.hexPosition(size=size)
        self.dir_path = os.path.join(os.getcwd(), args['model_output_folder'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded_models = []
        self.models = []
        self.args = args

    def load_models(self):
        model_files = [f for f in listdir(self.dir_path) if
                       isfile(os.path.join(self.dir_path, f)) and 'model_' in f and f not in self.loaded_models]
        model_files.sort(key=lambda e: int(e.split('_')[1].split('.')[0]))
        for model_name in model_files:
            model = ResNet(self.game, 9, 128, self.device)
            model.load_state_dict(torch.load(os.path.join(self.dir_path, model_name), map_location=self.device))
            model.eval()
            self.models.append(model)
        if len(model_files) != 0:
            self.loaded_models += model_files

    @torch.no_grad()
    def model_vs_model(self, model1, model2, iterations):
        model1_model2 = [0, 0]

        for i in range(iterations):
            self.game.reset()
            while True:
                if self.game.player == 1:
                    encoded_state = get_encoded_state(get_state(self.game))
                    policy, _ = model1(torch.tensor(encoded_state, device=model1.device).unsqueeze(0))
                    action_probs = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = get_valid_moves(encoded_state)
                    action_probs *= valid_moves
                    action = self.game.scalar_to_coordinates(np.argmax(action_probs))
                else:
                    self.game.board = self.game.recode_black_as_white()
                    self.game.player = 1
                    encoded_state = get_encoded_state(get_state(self.game))
                    policy, _ = model2(torch.tensor(encoded_state, device=model2.device).unsqueeze(0))
                    action_probs = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = get_valid_moves(encoded_state)
                    action_probs *= valid_moves
                    action = self.game.recode_coordinates(self.game.scalar_to_coordinates(np.argmax(action_probs)))
                    self.game.board = self.game.recode_black_as_white()
                    self.game.player = -1

                self.game.moove(action)
                if self.game.winner != 0 or len(self.game.get_action_space()) == 0:
                    model1_model2[0 if self.game.winner == 1 else 1] += 1
                    break
        return model1_model2

    @torch.no_grad()
    def model_vs_random(self, model, iterations):
        model_random = [0, 0]

        for i in range(iterations):
            self.game.reset()
            while True:
                if self.game.player == 1:
                    encoded_state = get_encoded_state(get_state(self.game))
                    policy, _ = model(torch.tensor(encoded_state, device=model.device).unsqueeze(0))
                    action_probs = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = get_valid_moves(encoded_state)
                    action_probs *= valid_moves
                    action = self.game.scalar_to_coordinates(np.argmax(action_probs))
                else:
                    action = choice(self.game.get_action_space())

                self.game.moove(action)
                if self.game.winner != 0 or len(self.game.get_action_space()) == 0:
                    model_random[0 if self.game.winner == 1 else 1] += 1
                    break
        return model_random

    @torch.no_grad()
    def model_vs_model_mem(self, model1, model2, iterations):
        return_memory = []
        model1_model2 = [0, 0]

        for i in range(iterations):
            self.game.reset()
            memory = []
            while True:
                if self.game.player == 1:
                    encoded_state = get_encoded_state(get_state(self.game))
                    policy1, _ = model1(torch.tensor(encoded_state, device=model1.device).unsqueeze(0))
                    action_probs1 = torch.softmax(policy1, dim=1).squeeze(0).cpu().numpy()
                    policy2, _ = model2(torch.tensor(encoded_state, device=model2.device).unsqueeze(0))
                    action_probs2 = torch.softmax(policy2, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = get_valid_moves(encoded_state)
                    action_probs1 *= valid_moves
                    action_probs2 *= valid_moves
                    action = self.game.scalar_to_coordinates(np.argmax(action_probs1))
                else:
                    self.game.board = self.game.recode_black_as_white()
                    self.game.player = 1
                    encoded_state = get_encoded_state(get_state(self.game))
                    policy1, _ = model1(torch.tensor(encoded_state, device=model1.device).unsqueeze(0))
                    action_probs1 = torch.softmax(policy1, dim=1).squeeze(0).cpu().numpy()
                    policy2, _ = model2(torch.tensor(encoded_state, device=model2.device).unsqueeze(0))
                    action_probs2 = torch.softmax(policy2, dim=1).squeeze(0).cpu().numpy()
                    valid_moves = get_valid_moves(encoded_state)
                    action_probs1 *= valid_moves
                    action_probs2 *= valid_moves
                    action = self.game.recode_coordinates(self.game.scalar_to_coordinates(np.argmax(action_probs2)))
                    self.game.board = self.game.recode_black_as_white()
                    self.game.player = -1

                memory.append((get_state(self.game), (action_probs1, action_probs2)))
                self.game.moove(action)
                if self.game.winner != 0 or len(self.game.get_action_space()) == 0:
                    model1_model2[0 if self.game.winner == 1 else 1] += 1
                    for hist_game_state, hist_action_probs in memory:
                        action_probs = hist_action_probs[0] if self.game.winner == 1 else hist_action_probs[1]
                        return_memory.append((
                            get_encoded_state(hist_game_state),
                            action_probs,
                            self.game.winner
                        ))
                    break
        return model1_model2, return_memory


if __name__ == "__main__":
    args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 1000,
        'num_selfPlay_iterations': 100,
        'num_parallel_games': 10,
        'num_epochs': 4,
        'batch_size': 10,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3,
        'num_eval_games': 200,
        'game_size': 7
    }

    eval = Eval(args)
    eval.load_models()
    print(eval.model_vs_model(eval.models[0], eval.models[1]))
    print(eval.model_vs_random(eval.models[0]))