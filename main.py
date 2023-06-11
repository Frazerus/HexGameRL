from AlphaZero import AlphaZero
import torch
import random
import numpy as np
from Model import ResNet
from engine import hex_engine
from Eval import Eval
import os

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


args = {'C': 2, 'num_searches': 1000, 'num_iterations': 100, 'iteration_offset': 0, 'num_selfPlay_iterations': 10, 'num_parallel_games': 2, 'num_epochs': 4, 'batch_size': 32, 'temperature': 1.25, 'dirichlet_epsilon': 0.25, 'dirichlet_alpha': 0.3, 'num_eval_games': 1000, 'game_size': 7, 'reinforce': False, 'num_reinforce_epochs': 4, 'model_output_folder': 'models', 'result_output_file': 'results.csv', 'time_limit_s': 1728000}


print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
game = hex_engine.hexPosition(size=args['game_size'])
model = ResNet(game, 9, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

if args['iteration_offset'] != 0:
    model.load_state_dict(torch.load(os.path.join(args['model_output_folder'], 'model_' + str(args['iteration_offset']) + '.pt'), map_location=device))
    optimizer.load_state_dict(torch.load(os.path.join(args['model_output_folder'], 'optimizer_' + str(args['iteration_offset']) + '.pt')))

if args['iteration_offset'] == 0:
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
