"""
snake_game.py – Core 20x20 Snake logic (no display).

State tensor returned by get_state():
  shape (20, 20, 3)  float32
  channel 0 = snake body
  channel 1 = snake head
  channel 2 = fruit

Extra state tracked per episode (used by reward functions in rewards.py):
  game.steps_since_fruit  – steps since last fruit eaten
  game.visited_cells      – set of (r,c) cells visited this episode
"""
import numpy as np
from collections import deque
import random

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DIRS = {
    UP:    (-1,  0),
    DOWN:  ( 1,  0),
    LEFT:  ( 0, -1),
    RIGHT: ( 0,  1),
}

OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


class SnakeGame:
    def __init__(self, size: int = 20, reward_fn=None, fruit_timeout: int = None):
        """
        size          : grid dimension (default 20x20)
        reward_fn     : callable(game, ate_fruit, dead) -> float
                        If None, uses the basic +10/-10/-0.01 reward.
        fruit_timeout : if set, episode ends if fruit is not reached within
                        this many steps (returns reward -15, done=True).
                        None = no timeout enforced at game level.
        """
        self.size          = size
        self.reward_fn     = reward_fn
        self.fruit_timeout = fruit_timeout
        self.reset()

    # ── init ─────────────────────────────────────

    def reset(self):
        mid = self.size // 2
        self.snake           = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.direction       = RIGHT
        self.score           = 0
        self.done            = False
        self.steps           = 0
        self.steps_since_fruit = 0
        self.visited_cells   = set(self.snake)   # cells seen this episode
        self._prev_dist      = None              # used by some reward functions
        self._spawn_fruit()
        return self.get_state()

    def _spawn_fruit(self):
        occupied = set(self.snake)
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in occupied
        ]
        self.fruit = random.choice(free) if free else None

    # ── step ─────────────────────────────────────

    def step(self, action: int):
        """
        Apply action (UP/DOWN/LEFT/RIGHT).
        Returns (state, reward, done).

        Reward comes from self.reward_fn if set, otherwise basic +10/-10/-0.01.
        """
        if self.done:
            return self.get_state(), 0.0, True

        # Ignore 180-degree reversals
        if action != OPPOSITE[self.direction]:
            self.direction = action

        dr, dc = DIRS[self.direction]
        hr, hc = self.snake[0]
        nr, nc = hr + dr, hc + dc

        # ── wall collision ───────────────────────
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            self.done = True
            reward = self.reward_fn(self, False, True) if self.reward_fn else -10.0
            return self.get_state(), reward, True

        # ── self collision ───────────────────────
        if (nr, nc) in list(self.snake)[:-1]:
            self.done = True
            reward = self.reward_fn(self, False, True) if self.reward_fn else -10.0
            return self.get_state(), reward, True

        # ── fruit timeout check ──────────────────
        if self.fruit_timeout and self.steps_since_fruit >= self.fruit_timeout:
            self.done = True
            return self.get_state(), -15.0, True

        # ── move ────────────────────────────────
        self.snake.appendleft((nr, nc))
        ate_fruit = (nr, nc) == self.fruit

        if ate_fruit:
            self.score           += 1
            self.steps_since_fruit = 0
            self.visited_cells   = set(self.snake)  # reset visited on new fruit
            self._spawn_fruit()
        else:
            self.snake.pop()
            self.steps_since_fruit += 1

        self.steps += 1

        # ── reward ──────────────────────────────
        if self.reward_fn:
            reward = self.reward_fn(self, ate_fruit, False)
        else:
            reward = 10.0 if ate_fruit else -0.01

        return self.get_state(), reward, False

    # ── state ────────────────────────────────────

    def get_state(self) -> np.ndarray:
        """
        Returns float32 array (size, size, 3):
          ch 0 – body  (1.0 where snake is)
          ch 1 – head  (1.0 at head)
          ch 2 – fruit (1.0 at fruit)
        """
        grid = np.zeros((self.size, self.size, 3), dtype=np.float32)
        for r, c in self.snake:
            grid[r, c, 0] = 1.0
        hr, hc = self.snake[0]
        grid[hr, hc, 1] = 1.0
        if self.fruit:
            grid[self.fruit[0], self.fruit[1], 2] = 1.0
        return grid

    def get_adjacent(self, pos=None) -> dict:
        """
        {action: cell_type} for 4 neighbours of pos (default: head).
        cell_type in {'empty', 'body', 'wall', 'fruit'}
        """
        r, c     = pos if pos is not None else self.snake[0]
        occupied = set(self.snake)
        result   = {}
        for action, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                result[action] = "wall"
            elif (nr, nc) == self.fruit:
                result[action] = "fruit"
            elif (nr, nc) in occupied:
                result[action] = "body"
            else:
                result[action] = "empty"
        return result

    def is_safe_move(self, action: int) -> bool:
        """True if action does not immediately kill the snake."""
        dr, dc = DIRS[action]
        hr, hc = self.snake[0]
        nr, nc = hr + dr, hc + dc
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return False
        if (nr, nc) in list(self.snake)[:-1]:
            return False
        return True
