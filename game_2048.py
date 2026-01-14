"""
Simple 2048 environment coded to be used for reinforcement learning.

This implementation uses a 4×4 NumPy array to store the board.  Action
space: 0=up, 1=down, 2=left, 3=right.  When equal tiles collide they
merge and generate a reward equal to the merged value.  After each
successful move a new tile (2 with 90 % chance or 4 with 10 %) is
added at a random empty. The game ends when no moves are possible.
These follow the official rules of the 2048 game and this code is the 
foundation for testing certain algorithms to 'solve' 2048.

"""

import random
from typing import Tuple
import numpy as np

class Game2048:
    def __init__(self) -> None:
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.action_map = {
            0: self._move_up,
            1: self._move_down,
            2: self._move_left,
            3: self._move_right,
        }

    def reset(self) -> np.ndarray:
        self.board[:] = 0
        self.score = 0
        for _ in range(2):
            self._spawn_tile()
        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        if action not in self.action_map:
            raise ValueError("Action must be 0 (up), 1 (down), 2 (left) or 3 (right)")
        prev_board = self.board.copy()
        reward = self.action_map[action]()  # perform move and compute reward
        if not np.array_equal(prev_board, self.board):
            self._spawn_tile()
        self.score += reward
        done = not self._can_move()
        return self.board.copy(), reward, done

    # --- Internal helpers ---

    def _spawn_tile(self) -> None:
        empty = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if not empty:
            return
        i, j = random.choice(empty)
        self.board[i, j] = 4 if random.random() < 0.1 else 2

    def _can_move(self) -> bool:
        if np.any(self.board == 0):
            return True
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1] or self.board[j, i] == self.board[j + 1, i]:
                    return True
        return False

    def _move_left(self) -> int:
        reward = 0
        for i in range(4):
            row = self.board[i, :].copy()
            filtered = row[row != 0]
            new_row = []
            j = 0
            while j < len(filtered):
                if j + 1 < len(filtered) and filtered[j] == filtered[j + 1]:
                    merged = filtered[j] * 2
                    reward += merged
                    new_row.append(merged)
                    j += 2
                else:
                    new_row.append(filtered[j])
                    j += 1
            new_row.extend([0] * (4 - len(new_row)))
            self.board[i, :] = new_row
        return reward

    def _move_right(self) -> int:
        self.board = np.fliplr(self.board)
        reward = self._move_left()
        self.board = np.fliplr(self.board)
        return reward

    def _move_up(self) -> int:
        self.board = self.board.T
        reward = self._move_left()
        self.board = self.board.T
        return reward

    def _move_down(self) -> int:
        self.board = self.board.T
        reward = self._move_right()
        self.board = self.board.T
        return reward


def play_random_games(num_games: int = 1, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    for idx in range(num_games):
        env = Game2048()
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = np.random.choice(4)
            state, reward, done = env.step(int(action))
            total_reward += reward
            steps += 1
        print(f"Game {idx+1}: finished in {steps} moves. Score: {total_reward}")

if __name__ == "__main__":
    play_random_games(num_games=10)
