"""
agent.py – SnakeAgent: wires the HomemadeCNN to the game.

Decision pipeline (every frame):
  1. get_state()         -> 20x20x3 feature map
  2. CNN.forward()       -> 4 raw logits  (one per direction)
  3. score_function()    -> heuristic scores (per direction)
  4. loop_penalty()      -> penalise revisiting recent positions
  5. Mask unsafe moves   -> -inf for wall / self-collision
  6. argmax(combined)    -> final action

──────────────────────────────────────────────────────────────────
SCORE FUNCTIONS  (all in agent.py, all visible here)
──────────────────────────────────────────────────────────────────

  score_fruit_distance   Rewards moving closer to fruit each step.
                         Returns a value in [-1, 0]: closer = less negative.
                         Purely positive incentive — never punishes distance.

  score_open_neighbors   Counts free cells around the NEXT position (1-step lookahead).
                         Returns [0, 1]: more open = higher.
                         Prevents stepping into dead ends.

  score_loop_penalty     Checks the last LOOP_MEMORY head positions.
                         Returns a large negative if the next cell was recently visited.
                         Breaks the circling pattern without retraining.

  score_function()       Combines the above with tunable weights w_dist, w_open.
                         Called every step for all 4 directions.

──────────────────────────────────────────────────────────────────
TUNING
──────────────────────────────────────────────────────────────────
All tunable constants are at the top of each function with comments.

In SnakeAgent.__init__:
  cnn_w    – how much the CNN logits contribute to the final decision
  score_w  – how much the heuristic (score_function) contributes
  w_dist   – weight of fruit-distance component inside score_function
  w_open   – weight of open-neighbors component inside score_function
  loop_w   – weight of loop-penalty component (set 0 to disable)
  loop_mem – how many recent head positions to remember for loop detection
"""
import numpy as np
from collections import deque
from snake_game import SnakeGame, UP, DOWN, LEFT, RIGHT, DIRS
from cnn import HomemadeCNN, softmax

CELL_VAL = {"empty": 0.5, "fruit": 1.0, "body": -1.0, "wall": -1.0}
ACTIONS   = (UP, DOWN, LEFT, RIGHT)


# ─────────────────────────────────────────────────
#  Score functions  (edit these to shape behaviour)
# ─────────────────────────────────────────────────

def score_fruit_distance(game: SnakeGame, action: int) -> float:
    """
    Rewards the next position being closer to fruit.

    Returns value in [-1, 0]:
      -0.0  = already on fruit (won't happen, but conceptually)
      -1.0  = furthest possible from fruit

    This is a POSITIVE incentive only — moving away gives 0 change
    in score, not a penalty. The fruit is always attractive.

    Tune: nothing to tune here — the weighting is done via w_dist
    in score_function().
    """
    if not game.fruit:
        return 0.0
    dr, dc = DIRS[action]
    hr, hc = game.snake[0]
    nr, nc = hr + dr, hc + dc
    fr, fc = game.fruit
    dist   = abs(nr - fr) + abs(nc - fc)
    return -dist / (2.0 * game.size)          # normalised [-1, 0], closer = higher


def score_open_neighbors(game: SnakeGame, action: int) -> float:
    """
    Counts how many of the 4 cells around the NEXT position are free.
    Returns [0, 1]: 1.0 = all 4 neighbors free, 0.0 = completely boxed in.

    1-step lookahead only. For deeper lookahead, see rewards.py flood fill.

    Tune: nothing to tune here — the weighting is done via w_open
    in score_function().
    """
    dr, dc   = DIRS[action]
    hr, hc   = game.snake[0]
    nr, nc   = hr + dr, hc + dc
    occupied = set(game.snake)
    free     = 0
    for a_dr, a_dc in DIRS.values():
        xr, xc = nr + a_dr, nc + a_dc
        if (0 <= xr < game.size and 0 <= xc < game.size
                and (xr, xc) not in occupied):
            free += 1
    return free / 4.0


def score_loop_penalty(game: SnakeGame, action: int,
                       recent_heads: deque) -> float:
    """
    Penalises moving into a cell that was recently a head position.

    This directly breaks the circling/looping pattern.
    The snake has no memory in its CNN input — it can't see where it
    has recently been. This function adds that memory at decision time.

    How it works:
      - The agent maintains a deque of the last LOOP_MEMORY head positions.
      - If the next cell appears in that deque, return LOOP_PENALTY.
      - The penalty scales with how recently the cell was visited:
          very recent visit = full penalty
          older visit = smaller penalty

    Tune:
      LOOP_PENALTY  – base penalty for revisiting a recent cell.
                      Should be large enough to override CNN + heuristic.
                      Set to 0 to disable loop breaking entirely.
    """
    LOOP_PENALTY = -2.0

    dr, dc = DIRS[action]
    hr, hc = game.snake[0]
    nr, nc = hr + dr, hc + dc
    next_pos = (nr, nc)

    if not recent_heads:
        return 0.0

    # Count how many times this cell appears in recent history
    # More recent = heavier weight via position in deque
    mem = list(recent_heads)
    penalty = 0.0
    n = len(mem)
    for i, pos in enumerate(reversed(mem)):   # most recent first
        if pos == next_pos:
            # weight decays the older the visit: most recent = full, oldest = 0.1
            weight = 1.0 - (i / n) * 0.9
            penalty += LOOP_PENALTY * weight
    return penalty


def score_function(game: SnakeGame, action: int,
                   w_dist: float = 1.0, w_open: float = 0.5) -> float:
    """
    Combined heuristic score for one action.
    Higher = better.

    Components:
      w_dist * score_fruit_distance  – pull toward fruit
      w_open * score_open_neighbors  – avoid dead ends

    The loop penalty is handled separately in decide() because it
    needs the agent's recent_heads state.

    Tune w_dist and w_open in SnakeAgent.__init__ or per-experiment
    in experiment.py ExperimentConfig.
    """
    return (w_dist * score_fruit_distance(game, action)
          + w_open * score_open_neighbors(game, action))


# ─────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────

class SnakeAgent:
    """
    Parameters
    ----------
    cnn      : pre-built HomemadeCNN (one is created if None)
    arch     : layer config list forwarded to HomemadeCNN
    cnn_w    : weight of CNN logits in final combined score
    score_w  : weight of heuristic (score_function) in final score
    w_dist   : fruit-distance weight inside score_function
    w_open   : open-neighbors weight inside score_function
    loop_w   : loop-penalty weight  (0 = disabled)
    loop_mem : number of recent head positions to remember
    """

    def __init__(self, cnn: HomemadeCNN = None, arch=None,
                 cnn_w:   float = 1.0,
                 score_w: float = 0.4,
                 w_dist:  float = 1.5,
                 w_open:  float = 0.5,
                 loop_w:  float = 1.0,
                 loop_mem: int  = 20):
        self.cnn      = cnn or HomemadeCNN(arch=arch)
        self.cnn_w    = cnn_w
        self.score_w  = score_w
        self.w_dist   = w_dist
        self.w_open   = w_open
        self.loop_w   = loop_w
        self.recent_heads = deque(maxlen=loop_mem)
        self._last_steps  = -1    # detects game reset

    def _check_reset(self, game: SnakeGame):
        """Clear loop memory when a new episode starts."""
        if game.steps < self._last_steps:
            self.recent_heads.clear()
        self._last_steps = game.steps

    # ── decision (called every frame) ────────────

    def decide(self, game: SnakeGame):
        """
        Full decision pipeline. Returns:
          action      : int  chosen direction
          final_probs : (4,) softmax of combined scores
          cnn_probs   : (4,) softmax of raw CNN logits (for display)
        """
        self._check_reset(game)

        state      = game.get_state()
        cnn_logits = self.cnn.forward(state)
        cnn_probs  = softmax(cnn_logits)

        combined = np.zeros(4, dtype=np.float32)
        for a in ACTIONS:
            if not game.is_safe_move(a):
                combined[a] = -1e9
            else:
                combined[a] = (
                    self.cnn_w   * cnn_logits[a]
                  + self.score_w * score_function(game, a, self.w_dist, self.w_open)
                  + self.loop_w  * score_loop_penalty(game, a, self.recent_heads)
                )

        final_probs = softmax(combined)
        action      = int(np.argmax(combined))

        # Record head position AFTER decision (before move happens)
        self.recent_heads.append(game.snake[0])

        return action, final_probs, cnn_probs
