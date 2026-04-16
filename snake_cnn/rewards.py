"""
rewards.py  –  Swappable reward functions for SnakeGame.

Each function signature:
    fn(game, ate_fruit: bool, dead: bool) -> float

`game` is the SnakeGame instance AFTER the move has been applied,
so game.snake[0] is already the new head position.

Extra state available on the game after the snake_game.py update:
    game.steps_since_fruit   – steps since last fruit was eaten
    game.visited_cells       – set of (r,c) visited this episode

──────────────────────────────────────────────────────────────────
FUNCTION SUMMARY
──────────────────────────────────────────────────────────────────

  reward_basic          Original: +10 fruit, -10 death, -0.01 step.
                        The one that taught "fruit = bad".

  reward_efficiency     Fruit bonus decays with steps taken.
                        Not finding fruit in TIMEOUT steps = worse than death.
                        Distance gradient pushes head toward fruit every step.

  reward_no_repeat      Rewards visiting new cells, penalises looping back.
                        Keeps the snake exploring rather than circling.
                        NOTE: visited_cells is not a CNN input channel yet
                        so this signal only shapes fitness, not per-step decisions.

  reward_flood_safety   Flood-fill from head after each step.
                        Rewards keeping a large reachable area.
                        Penalises moves that cut off big sections of the board.
                        Expensive (~400 BFS ops/step) but the most "aware" signal.

  reward_combined       All of the above in one function.
                        Recommended starting point for real training.

──────────────────────────────────────────────────────────────────
TUNING GUIDE
──────────────────────────────────────────────────────────────────
Change the constants at the top of each function.  The comments
explain what each constant does.

To register a new function for experiments, add it to REWARD_FNS
at the bottom of this file.
"""
from collections import deque


# ─────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────

def _dist_to_fruit(game) -> float:
    """Manhattan distance from current head to fruit.  0 if no fruit."""
    if not game.fruit:
        return 0.0
    hr, hc = game.snake[0]
    return float(abs(hr - game.fruit[0]) + abs(hc - game.fruit[1]))


def _flood_fill_count(game) -> int:
    """
    BFS from snake head counting reachable empty cells.
    O(grid_size^2) worst case (~400 ops on 20x20).
    """
    head     = game.snake[0]
    occupied = set(game.snake)
    visited  = {head}
    q        = deque([head])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < game.size and 0 <= nc < game.size
                    and (nr, nc) not in occupied
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                q.append((nr, nc))
    return len(visited)


# ─────────────────────────────────────────────────
#  1. Basic (original)
# ─────────────────────────────────────────────────

def reward_basic(game, ate_fruit: bool, dead: bool) -> float:
    """
    Original reward.  Simple but problematic:
    eating fruit grows the snake -> harder to survive -> CNN learns to avoid fruit.
    """
    if dead:      return -10.0
    if ate_fruit: return  10.0
    return -0.01


# ─────────────────────────────────────────────────
#  2. Efficiency  (fruit-seeking + timeout)
# ─────────────────────────────────────────────────

def reward_efficiency(game, ate_fruit: bool, dead: bool) -> float:
    """
    Two key ideas:
      A) Not reaching fruit within TIMEOUT steps costs MORE than dying.
         This forces the snake to always be working toward fruit.
      B) Reaching fruit EARLY gives a large bonus that fades linearly.
         Optimal route = max bonus.  Wandering = no bonus.
      C) Each step gives a small gradient reward for moving closer.

    Tune:
      TIMEOUT        – steps allowed per fruit before the big penalty kicks in
      FRUIT_BASE     – base reward for eating fruit
      EARLY_BONUS    – extra reward if fruit eaten on step 1 (fades to 0 at TIMEOUT)
      TIMEOUT_PENALTY– cost for failing the timeout (should be > death penalty)
      CLOSER_REWARD  – per-step reward for reducing distance to fruit
      FARTHER_COST   – per-step cost for increasing distance
    """
    TIMEOUT         = 40
    FRUIT_BASE      = 10.0
    EARLY_BONUS     = 20.0   # max extra for instant fruit (step 1)
    TIMEOUT_PENALTY = -15.0  # worse than -10 death
    CLOSER_REWARD   =  0.3
    FARTHER_COST    = -0.2

    if dead:
        return -10.0

    if ate_fruit:
        steps = game.steps_since_fruit           # steps taken to reach this fruit
        early = max(0.0, 1.0 - steps / TIMEOUT)  # 1.0 at step 0, 0.0 at TIMEOUT
        return FRUIT_BASE + EARLY_BONUS * early

    # Timeout check — fruit not reached in time
    if game.steps_since_fruit >= TIMEOUT:
        return TIMEOUT_PENALTY

    # Distance gradient — reward/penalise each step based on direction
    dist_now  = _dist_to_fruit(game)
    dist_prev = game._prev_dist if game._prev_dist is not None else dist_now
    game._prev_dist = dist_now

    if dist_now < dist_prev:
        return CLOSER_REWARD
    elif dist_now > dist_prev:
        return FARTHER_COST
    return -0.01   # same distance, neutral


# ─────────────────────────────────────────────────
#  3. No-repeat  (penalise revisiting cells)
# ─────────────────────────────────────────────────

def reward_no_repeat(game, ate_fruit: bool, dead: bool) -> float:
    """
    Penalises the snake for entering a cell it has already visited this episode.
    Prevents the looping pattern where the snake circles endlessly avoiding fruit.

    Note: visited_cells only affects the fitness signal.
    The CNN doesn't see visited_cells as a channel by default.
    Add it as a 4th channel in snake_game.get_state() if you want the CNN
    to make per-step decisions based on it.

    Tune:
      SURVIVAL_PER_STEP – small positive to reward staying alive longer
      NEW_CELL_BONUS    – reward for visiting a cell not seen this episode
      REVISIT_PENALTY   – cost for stepping on an already-visited cell
      FRUIT_REWARD      – base fruit reward
      DEATH_PENALTY     – collision cost
    """
    SURVIVAL_PER_STEP =  0.05
    NEW_CELL_BONUS    =  0.2
    REVISIT_PENALTY   = -0.4
    FRUIT_REWARD      =  10.0
    DEATH_PENALTY     = -10.0

    if dead:
        return DEATH_PENALTY

    head = game.snake[0]

    if ate_fruit:
        game.visited_cells.add(head)
        return FRUIT_REWARD

    # cell novelty
    if head in game.visited_cells:
        cell_reward = REVISIT_PENALTY
    else:
        game.visited_cells.add(head)
        cell_reward = NEW_CELL_BONUS

    return SURVIVAL_PER_STEP + cell_reward


# ─────────────────────────────────────────────────
#  4. Flood safety  (keep open space, avoid traps)
# ─────────────────────────────────────────────────

def reward_flood_safety(game, ate_fruit: bool, dead: bool) -> float:
    """
    After each step, BFS-flood from the head and count reachable cells.
    Rewards moves that keep a large open area accessible.
    Penalises moves that cut off big chunks of the board.

    This addresses the core long-snake problem: as the body grows, many
    moves lead to enclosed areas.  The flood fill scores them in advance.

    Tune:
      SAFETY_SCALE   – multiplier on the open-space fraction [0,1]
      FRUIT_REWARD   – base fruit reward
      DEATH_PENALTY  – collision cost
    """
    SAFETY_SCALE  = 2.0
    FRUIT_REWARD  = 10.0
    DEATH_PENALTY = -10.0

    if dead:
        return DEATH_PENALTY

    reachable   = _flood_fill_count(game)
    total_empty = game.size * game.size - len(game.snake)
    safety      = reachable / max(1, total_empty)   # fraction [0, 1]

    if ate_fruit:
        return FRUIT_REWARD + SAFETY_SCALE * safety

    return SAFETY_SCALE * safety - 0.5   # offset so neutral step ~ 0


# ─────────────────────────────────────────────────
#  5. Combined  (recommended)
# ─────────────────────────────────────────────────

def reward_combined(game, ate_fruit: bool, dead: bool) -> float:
    """
    Combines all signals:
      – efficiency  : fruit-seeking with timeout and early bonus
      – no-repeat   : penalise looping / revisiting
      – flood safety: avoid cutting off open space

    Each component is weighted.  Edit the weights here to shift emphasis.

    Tune:
      W_EFFICIENCY  – how much to weight the fruit-seeking gradient
      W_NO_REPEAT   – how much to penalise revisiting cells
      W_FLOOD       – how much to reward open space
      (death and fruit rewards are applied once and not mixed)
    """
    W_EFFICIENCY = 1.0
    W_NO_REPEAT  = 0.6
    W_FLOOD      = 0.8

    TIMEOUT         = 40
    FRUIT_BASE      = 10.0
    EARLY_BONUS     = 15.0
    TIMEOUT_PENALTY = -15.0
    DEATH_PENALTY   = -10.0

    if dead:
        return DEATH_PENALTY

    if ate_fruit:
        steps = game.steps_since_fruit
        early = max(0.0, 1.0 - steps / TIMEOUT)
        # reset distance tracker and visited set on fruit eat
        game._prev_dist    = _dist_to_fruit(game)
        game.visited_cells = set(game.snake)
        return FRUIT_BASE + EARLY_BONUS * early

    # ── timeout penalty ──────────────────────────
    if game.steps_since_fruit >= TIMEOUT:
        return TIMEOUT_PENALTY

    # ── efficiency component ─────────────────────
    dist_now  = _dist_to_fruit(game)
    dist_prev = game._prev_dist if game._prev_dist is not None else dist_now
    game._prev_dist = dist_now
    eff = 0.3 if dist_now < dist_prev else (-0.2 if dist_now > dist_prev else -0.01)

    # ── no-repeat component ──────────────────────
    head = game.snake[0]
    if head in game.visited_cells:
        rep = -0.4
    else:
        game.visited_cells.add(head)
        rep = 0.1

    # ── flood safety component ───────────────────
    reachable   = _flood_fill_count(game)
    total_empty = game.size * game.size - len(game.snake)
    safety      = reachable / max(1, total_empty)
    flood       = safety - 0.5   # centre around 0

    return (W_EFFICIENCY * eff
          + W_NO_REPEAT  * rep
          + W_FLOOD      * flood)


# ─────────────────────────────────────────────────
#  Registry  (name -> function for experiments)
# ─────────────────────────────────────────────────


# ─────────────────────────────────────────────────
#  6. Attract  (fruit = always positive, never a repulsor)
# ─────────────────────────────────────────────────

def reward_attract(game, ate_fruit: bool, dead: bool) -> float:
    """
    Fixes the repulsor problem in reward_efficiency.

    Root cause of repulsor:
      The timeout penalty fires while the snake is still trying to reach fruit.
      Those approaching states accumulate negative reward, so training learns
      that being-near-fruit = bad. The fruit becomes a thing to avoid.

    Fix:
      - ONLY reward getting closer (positive signal, never negative for distance).
      - Keep +10 fruit and -10 death exactly as in reward_basic.
      - No timeout penalty ever.
      - Add flood-fill safety so the snake doesn't trap itself while seeking.

    The snake now has exactly two clear incentives:
      1. Get closer to fruit each step  (+CLOSER)
      2. Keep board open               (+flood)
    And two hard stops:
      1. Don't die                     (-10)
      2. Don't grow and cut yourself off  (flood penalises this)

    Tune:
      CLOSER_REWARD  – reward per step the head moves closer to fruit
      SAFETY_SCALE   – multiplier on flood-fill open-space fraction
    """
    CLOSER_REWARD = 0.4
    SAFETY_SCALE  = 1.0

    if dead:
        return -10.0

    if ate_fruit:
        # Reset distance tracker so the next fruit starts clean
        game._prev_dist = _dist_to_fruit(game)
        return 10.0   # unchanged — the score mechanic is not touched

    # ── distance signal: only reward progress, never punish distance ──
    dist_now  = _dist_to_fruit(game)
    dist_prev = game._prev_dist if game._prev_dist is not None else dist_now
    game._prev_dist = dist_now

    closer = CLOSER_REWARD if dist_now < dist_prev else 0.0  # positive or zero only

    # ── flood safety: keep board open ────────────────────────────────
    reachable   = _flood_fill_count(game)
    total_empty = game.size * game.size - len(game.snake)
    safety      = reachable / max(1, total_empty)   # [0, 1]
    flood       = SAFETY_SCALE * safety - 0.5       # centre around 0

    return closer + flood


REWARD_FNS = {
    "basic":       reward_basic,
    "efficiency":  reward_efficiency,
    "no_repeat":   reward_no_repeat,
    "flood":       reward_flood_safety,
    "combined":    reward_combined,
    "attract":     reward_attract,
}
