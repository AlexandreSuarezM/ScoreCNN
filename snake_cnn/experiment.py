"""
experiment.py  –  Define, run, and compare Snake CNN experiments.

Each experiment is an ExperimentConfig.  Add yours to EXPERIMENTS at the
bottom of this file, then run:

    python experiment.py                   # run all experiments
    python experiment.py default_baseline  # run one by name
    python experiment.py --list            # print all registered names

Results are saved to  results/<name>.json  and  results/<name>_weights.npy
Run  python compare.py  afterwards to see a ranked comparison table.

-- How to copy weights between experiments --

    from experiment import copy_cnn_weights, NAMED_ARCHS
    from cnn import HomemadeCNN
    import numpy as np

    source = HomemadeCNN(arch=NAMED_ARCHS["default"])
    source.set_weights(np.load("results/default_baseline_weights.npy"))

    target = HomemadeCNN(arch=NAMED_ARCHS["default"])  # same arch required
    copy_cnn_weights(source, target)                    # copies weights in-place
"""
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cnn     import HomemadeCNN, DEFAULT_ARCH, ARCH_1CONV, ARCH_DEEP
from agent   import SnakeAgent
from snake_game import SnakeGame
from rewards import REWARD_FNS

# ── named arch registry ───────────────────────────
# Add new named architectures here.
# Keys are stored in JSON results (no Python callables in files).

NAMED_ARCHS: dict = {
    "default": DEFAULT_ARCH,   # 2 conv + 2 dense
    "1conv":   ARCH_1CONV,     # minimal
    "deep":    ARCH_DEEP,      # 4 layers with pooling

    # Custom example – 3 conv, bigger dense:
    "3conv": [
        {"type": "conv",    "out_ch": 8,  "kernel": 3},
        {"type": "conv",    "out_ch": 16, "kernel": 3},
        {"type": "conv",    "out_ch": 32, "kernel": 3},
        {"type": "pool",    "size": 2},
        {"type": "flatten"},
        {"type": "dense",   "out_size": 128},
        {"type": "dense",   "out_size": 4, "activation": None},
    ],
}


# ── experiment config ─────────────────────────────

@dataclass
class ExperimentConfig:
    """
    All parameters that define one experiment.

    arch_name  : key into NAMED_ARCHS
    cnn_w      : weight of CNN logit in the final decision score
    score_w    : weight of heuristic score in the final decision score
    w_dist     : heuristic – fruit-distance component weight
    w_open     : heuristic – open-neighbors component weight
    sigma      : evolutionary mutation noise (std dev)
    generations: how many (1+1)-ES generations to run
    episodes_per_eval : episodes averaged per fitness evaluation
    max_steps  : max steps per episode before ending it
    description: shown in comparison table
    """
    name:              str
    arch_name:         str   = "default"
    cnn_w:             float = 1.0
    score_w:           float = 0.4
    w_dist:            float = 1.0
    w_open:            float = 0.5
    sigma:             float = 0.05
    generations:       int   = 300
    episodes_per_eval: int   = 5
    max_steps:         int   = 600
    reward_fn:         str   = "basic"    # key into rewards.REWARD_FNS
    fruit_timeout:     int   = None       # None = no episode timeout for missing fruit
    score_fitness:     bool  = False      # True = use game.score as primary fitness
    description:       str   = ""


# ── weight copy utility ───────────────────────────

def copy_cnn_weights(source: HomemadeCNN, target: HomemadeCNN) -> None:
    """
    Copy weights from source into target in-place.
    Both CNNs must have identical architectures (same total parameter count).

    Raises ValueError if weight vector sizes do not match.
    Useful for seeding one experiment from another's trained weights.
    """
    src_w = source.get_weights()
    tgt_w = target.get_weights()
    if src_w.shape != tgt_w.shape:
        raise ValueError(
            f"Architecture mismatch: source has {src_w.size} params, "
            f"target has {tgt_w.size} params.  Use the same arch_name."
        )
    target.set_weights(src_w.copy())


# ── single-episode runner ─────────────────────────

def _run_episode(agent: SnakeAgent, game: SnakeGame, max_steps: int,
                 score_fitness: bool = False) -> float:
    """
    Run one episode and return fitness.

    score_fitness=False (default):
        Returns sum of per-step rewards.
        Problem: high step-count reward can dominate fruit reward,
        so the snake learns to circle without eating.

    score_fitness=True (recommended):
        Returns  game.score * 1000 + game.steps
        Eating 1 fruit (1000) always beats surviving all steps (600).
        The snake can never win by circling — it must eat to score.
    """
    game.reset()
    for _ in range(max_steps):
        if game.done:
            break
        action, _, _ = agent.decide(game)
        game.step(action)

    if score_fitness:
        # fruit eaten is the primary objective, survival is a tiebreaker
        return game.score * 1000 + game.steps
    else:
        # legacy: re-run to accumulate rewards (kept for backwards compat)
        game.reset()
        total = 0.0
        for _ in range(max_steps):
            if game.done:
                break
            action, _, _ = agent.decide(game)
            _, reward, _ = game.step(action)
            total += reward
        return total


def _evaluate(agent: SnakeAgent, episodes: int, max_steps: int,
              reward_fn=None, fruit_timeout=None,
              score_fitness: bool = False) -> float:
    game = SnakeGame(reward_fn=reward_fn, fruit_timeout=fruit_timeout)
    return sum(
        _run_episode(agent, game, max_steps, score_fitness)
        for _ in range(episodes)
    ) / episodes


# ── experiment runner ─────────────────────────────

RESULTS_DIR = Path("results")
PRINT_EVERY = 10


def run_experiment(cfg: ExperimentConfig) -> dict:
    """
    Run the full (1+1)-ES training loop for one ExperimentConfig.

    Returns the result dict (same content as the saved JSON).
    Also saves:
        results/<name>.json           – config + metrics + training curve
        results/<name>_weights.npy    – best evolved weights
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    arch      = NAMED_ARCHS[cfg.arch_name]
    reward_fn = REWARD_FNS.get(cfg.reward_fn)
    agent     = SnakeAgent(
        arch    = arch,
        cnn_w   = cfg.cnn_w,
        score_w = cfg.score_w,
        w_dist  = cfg.w_dist,
        w_open  = cfg.w_open,
    )

    best_w   = agent.cnn.get_weights().copy()
    best_fit = _evaluate(agent, cfg.episodes_per_eval, cfg.max_steps,
                         reward_fn, cfg.fruit_timeout, cfg.score_fitness)

    print(f"  Gen     0  |  fitness = {best_fit:8.3f}  (random weights)")

    training_curve = [{"generation": 0, "best_fitness": float(best_fit)}]
    t0 = time.perf_counter()

    for gen in range(1, cfg.generations + 1):
        noise = (np.random.randn(*best_w.shape) * cfg.sigma).astype(np.float32)
        new_w = best_w + noise
        agent.cnn.set_weights(new_w)

        fit = _evaluate(agent, cfg.episodes_per_eval, cfg.max_steps,
                        reward_fn, cfg.fruit_timeout, cfg.score_fitness)
        if fit >= best_fit:
            best_fit = fit
            best_w   = new_w.copy()
        else:
            agent.cnn.set_weights(best_w)

        if gen % PRINT_EVERY == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Gen {gen:5d}  |  best = {best_fit:8.3f}  |  {elapsed:.1f}s")
            training_curve.append({"generation": gen, "best_fitness": float(best_fit)})

    duration = time.perf_counter() - t0

    # Save weights
    weights_path = RESULTS_DIR / f"{cfg.name}_weights.npy"
    np.save(str(weights_path), best_w)

    # Assemble result dict
    result = {
        "experiment":    asdict(cfg),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 2),
        "training_curve": training_curve,
        "final_metrics": {
            "best_fitness":          float(best_fit),
            "final_generation":      cfg.generations,
            "total_episodes_approx": cfg.generations * cfg.episodes_per_eval,
        },
    }

    json_path = RESULTS_DIR / f"{cfg.name}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved -> {json_path}  |  weights -> {weights_path}")
    return result


# ── experiment registry ───────────────────────────
# Add your experiments here.  Each entry is one ExperimentConfig.
# Run  python experiment.py --list  to see them.

EXPERIMENTS: list = [

    ExperimentConfig(
        name        = "default_baseline",
        arch_name   = "default",
        description = "2-conv + 2-dense, balanced score weights",
    ),

    ExperimentConfig(
        name        = "1conv_minimal",
        arch_name   = "1conv",
        generations = 200,
        description = "Minimal 1-conv arch, fast to train",
    ),

    ExperimentConfig(
        name        = "deep_pooling",
        arch_name   = "deep",
        generations = 400,
        description = "4 layers with max-pool, more capacity",
    ),

    ExperimentConfig(
        name        = "chase_fruit",
        arch_name   = "default",
        w_dist      = 2.5,
        w_open      = 0.1,
        description = "Score heavily biased toward fruit distance",
    ),

    ExperimentConfig(
        name        = "survival_first",
        arch_name   = "default",
        w_dist      = 0.3,
        w_open      = 1.8,
        description = "Score biased toward open space (stay alive)",
    ),

    ExperimentConfig(
        name        = "cnn_dominant",
        arch_name   = "default",
        cnn_w       = 3.0,
        score_w     = 0.1,
        description = "CNN logits dominate the decision (heuristic barely used)",
    ),

    ExperimentConfig(
        name        = "heuristic_only",
        arch_name   = "default",
        cnn_w       = 0.0,
        score_w     = 1.0,
        description = "Pure heuristic – CNN is ignored (control experiment)",
    ),

    ExperimentConfig(
        name        = "3conv_deep",
        arch_name   = "3conv",
        generations = 400,
        sigma       = 0.03,
        description = "3 conv layers + pooling, tighter mutation",
    ),

    ExperimentConfig(
        name        = "fast_mutation",
        arch_name   = "default",
        sigma       = 0.15,
        description = "Large mutation noise – explores faster, less stable",
    ),

    ExperimentConfig(
        name        = "fine_mutation",
        arch_name   = "default",
        sigma       = 0.01,
        generations = 500,
        description = "Tiny mutation – slow but precise refinement",
    ),

    # ── new reward function experiments ──────────

    ExperimentConfig(
        name          = "reward_efficiency",
        arch_name     = "default",
        reward_fn     = "efficiency",
        fruit_timeout = 40,
        generations   = 300,
        description   = "Timeout 40 steps/fruit. Early fruit = big bonus. Late = worse than death",
    ),

    ExperimentConfig(
        name          = "reward_no_repeat",
        arch_name     = "default",
        reward_fn     = "no_repeat",
        generations   = 300,
        description   = "Penalise revisiting cells. Breaks looping patterns",
    ),

    ExperimentConfig(
        name          = "reward_flood",
        arch_name     = "default",
        reward_fn     = "flood",
        generations   = 300,
        description   = "Flood-fill safety: rewards keeping open board area",
    ),

    ExperimentConfig(
        name          = "reward_combined",
        arch_name     = "default",
        reward_fn     = "combined",
        fruit_timeout = 40,
        generations   = 300,
        description   = "All signals: efficiency + no-repeat + flood safety",
    ),

    ExperimentConfig(
        name          = "reward_combined_deep",
        arch_name     = "deep",
        reward_fn     = "combined",
        fruit_timeout = 40,
        generations   = 400,
        sigma         = 0.04,
        description   = "Combined reward + deeper arch + tighter mutation",
    ),

    ExperimentConfig(
        name        = "reward_attract",
        arch_name   = "default",
        reward_fn   = "attract",
        generations = 300,
        description = "Fruit = always positive. Closer=reward, never penalise distance",
    ),

    ExperimentConfig(
        name        = "reward_attract_deep",
        arch_name   = "deep",
        reward_fn   = "attract",
        generations = 400,
        sigma       = 0.04,
        description = "Attract reward + deeper arch",
    ),

    # ── score-fitness experiments (fruit = primary objective) ─────────
    # score_fitness=True means fitness = game.score*1000 + steps
    # Circling can never beat eating. Evolution MUST learn to eat.

    ExperimentConfig(
        name          = "score_attract",
        arch_name     = "default",
        reward_fn     = "attract",
        score_fitness = True,
        fruit_timeout = 60,
        generations   = 300,
        description   = "Fitness=fruits*1000+steps. Attract reward. Circling = losing",
    ),

    ExperimentConfig(
        name          = "score_flood",
        arch_name     = "default",
        reward_fn     = "flood",
        score_fitness = True,
        fruit_timeout = 60,
        generations   = 300,
        description   = "Fitness=fruits*1000+steps. Flood safety. Must eat to score",
    ),

    ExperimentConfig(
        name          = "score_attract_deep",
        arch_name     = "deep",
        reward_fn     = "attract",
        score_fitness = True,
        fruit_timeout = 60,
        generations   = 400,
        sigma         = 0.04,
        description   = "Score-fitness + attract + deep arch",
    ),
]


# ── CLI entry point ───────────────────────────────

def main():
    args = sys.argv[1:]

    if "--list" in args:
        print(f"\n{'Name':<25}  {'Arch':<10}  {'Gens':>5}  Description")
        print("-" * 70)
        for cfg in EXPERIMENTS:
            print(f"  {cfg.name:<23}  {cfg.arch_name:<10}  {cfg.generations:>5}  {cfg.description}")
        print()
        return

    targets = [c for c in EXPERIMENTS if not args or c.name in args]

    if not targets:
        print(f"No experiments matched: {args}")
        print("Run  python experiment.py --list  to see available names.")
        return

    total = len(targets)
    for i, cfg in enumerate(targets, 1):
        print(f"\n{'='*58}")
        print(f"  [{i}/{total}]  {cfg.name}")
        print(f"  arch={cfg.arch_name}  sigma={cfg.sigma}  "
              f"w_dist={cfg.w_dist}  w_open={cfg.w_open}")
        print(f"  {cfg.description}")
        print(f"{'='*58}")
        run_experiment(cfg)

    print(f"\nAll done.  Run  python compare.py  to see results.")


if __name__ == "__main__":
    main()
