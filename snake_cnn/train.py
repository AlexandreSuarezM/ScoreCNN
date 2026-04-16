"""
train.py – Evolutionary training for the Snake CNN agent.

Uses a simple (1+1)-Evolution Strategy:
  1. Evaluate current weights → fitness
  2. Mutate a copy with Gaussian noise (sigma)
  3. Keep the mutation if fitness improves
  4. Repeat

Fitness = average total reward over EPISODES_PER_EVAL episodes.

The score function in agent.py guides the network toward fruit while
avoiding walls, providing a shaped reward signal even with random weights.

Run:
    python train.py

Then in main.py, uncomment the weight-loading block to see the trained agent.
"""
import os
import time
import numpy as np

from snake_game import SnakeGame
from cnn        import HomemadeCNN, DEFAULT_ARCH
from agent      import SnakeAgent

# ── hyper-parameters ──────────────────────────────
EPISODES_PER_EVAL = 5       # episodes averaged per fitness evaluation
MAX_STEPS         = 600     # max steps before ending an episode
SIGMA             = 0.05    # mutation noise (Gaussian std dev)
GENERATIONS       = 300
PRINT_EVERY       = 10
SAVE_PATH         = "best_weights.npy"


def run_episode(agent: SnakeAgent, game: SnakeGame) -> float:
    """Run one episode and return total reward."""
    game.reset()
    total = 0.0
    for _ in range(MAX_STEPS):
        if game.done:
            break
        action, _, _ = agent.decide(game)
        _, reward, _ = game.step(action)
        total += reward
    return total


def evaluate(agent: SnakeAgent, episodes: int = EPISODES_PER_EVAL) -> float:
    """Average reward over multiple episodes."""
    game = SnakeGame()
    return sum(run_episode(agent, game) for _ in range(episodes)) / episodes


def main():
    print("─" * 50)
    print(" Snake CNN – Evolutionary Training")
    print(f" Architecture: {DEFAULT_ARCH}")
    print("─" * 50)

    agent  = SnakeAgent(arch=DEFAULT_ARCH)
    best_w = agent.cnn.get_weights().copy()

    print("Evaluating initial (random) weights…")
    best_fit = evaluate(agent)
    print(f"Gen    0  │  fitness = {best_fit:8.2f}\n")

    t0 = time.time()

    for gen in range(1, GENERATIONS + 1):
        # Mutate
        noise = (np.random.randn(*best_w.shape) * SIGMA).astype(np.float32)
        new_w = best_w + noise
        agent.cnn.set_weights(new_w)

        fit = evaluate(agent)
        if fit >= best_fit:           # accept if not worse (greedy)
            best_fit = fit
            best_w   = new_w.copy()
        else:
            agent.cnn.set_weights(best_w)

        if gen % PRINT_EVERY == 0:
            elapsed = time.time() - t0
            print(f"Gen {gen:5d}  │  best fitness = {best_fit:8.2f}"
                  f"  │  {elapsed:.1f}s")
            np.save(SAVE_PATH, best_w)

    np.save(SAVE_PATH, best_w)
    print(f"\nDone!  Best fitness: {best_fit:.2f}")
    print(f"Weights saved to '{SAVE_PATH}'")
    print("\nTo use in main.py, uncomment the weight-loading block.")


if __name__ == "__main__":
    main()
