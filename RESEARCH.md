# Research Notes

Technical documentation for the Goodhart's Law ML project.
This file grows as the project progresses — methodology, experiments, observations, results.

---

## Experiment 1 — Reproducing the failure mode

### Setup
- Environment: Snake (grid-based, standard rules)
- Agent: CNN on raw board state
- Reward function: +score for eating food, penalty for death, neutral for survival

### Hypothesis
The agent will learn a survival strategy that satisfies the reward function without learning to play effectively. Expected behaviour: looping patterns, avoidance of risk, stagnation of score.

### Results
*In progress.*

---

## Experiment 2 — Structured input representation

### Setup
Same environment and reward function. Input layer replaced with engineered features:

- Count of safe squares remaining
- Tail position and adjacent squares
- Danger ranking per square (proximity to tail, walls, dead ends)
- Distance and path accessibility to food

### Hypothesis
With structured input, the model has fewer degrees of freedom to exploit unintended shortcuts. The reward function does not change — only what the model sees changes.

### Results
*Pending experiment 1.*

---

## Observations

*Notes, unexpected findings, and open questions added here as the project develops.*

---

## On scoring function design

A secondary question this project explores: can scoring functions be learned rather than hand-designed?

If input representation reduces the search space for reward hacking, does it also make the scoring function easier to specify — or to learn from examples of intended behaviour?

*This is a later-stage question. Noted here for continuity.*
