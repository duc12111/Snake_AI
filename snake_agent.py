import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np
import torch

from snake_game import SnakeGameAI, Direction, Point
import argparse

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.0001


class Agent:
    def __init__(self, game: SnakeGameAI):
        self.number_games = 0
        self.epsilon = 0.3
        self.gamma = 0
        self.memory = deque(maxlen=args['max_memory'])
        self.game = game

        # TODO model trainer
        self.model = None
        self.trainer = None

    def get_state(self):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if self.epsilon > 0 and self.number_games % 50 == 49:
            self.epsilon -= 0.02
        action = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            prediction = self.model(torch.Tensor(state,dtype =torch.float))
            move = torch.argmax(prediction).item()
            action[move] = 1
            
        return action

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < args['batch']:
            samples = self.memory
        else:
            samples = random.sample(self.memory, args['batch'])
        states, action, rewards, next_states, dones = zip(*samples)
        self.trainer.train_step(states, action, rewards, next_states, dones)


def train():
    plot_scores = []
    plot_mean_scored = []
    total_score = 0
    record = 0
    game = SnakeGameAI()
    agent = Agent(game)
    while True:
        state_old = agent.get_state()

        action = agent.get_action(state_old)

        reward, done, score = game.self_play_step(action)

        state_new = agent.get_state()

        # Memory
        agent.train_short_memory(state_new, action, reward, state_new, done)

        agent.remember(state_new, action, reward, state_new, done)

        if done:
            game.reset()
            agent.number_games += 1
            self.train_long_memory()
            if score > record:
                record = score
                # agent.model.save()

            print(f"Game {self.number_games} : Score {score}, The current Record {record}")

            # TODO plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_memory', type=int, default=100000, help='Max_Memory')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning_Rate')
    parser.add_argument('--batch', type=int, default=1000, help='Batch_size')
    args = parser.parse_args()
