# Goodhart's Law — Metric Misalignment in Machine Learning

> *"When a measure becomes a target, it ceases to be a good measure."*
> — Charles Goodhart, 1975

---

## Why this matters

Machine learning systems are being deployed to make real decisions — in finance, hiring, healthcare, content moderation. In every case, the model optimises a metric. And in every case, that metric is a proxy for what we actually want, not the thing itself.

That gap is not a minor technical detail. It is where things go wrong.

When the proxy is good enough, the system works. When it isn't — or when it gets optimised hard enough that its weaknesses are exposed — the system produces results that satisfy the score while violating the intent. This happens quietly, at scale, and often goes unnoticed until the damage is done.

This project investigates that gap: where it comes from, why it persists, and whether it can be reduced through better design of what we feed our models.

---

## The law

Goodhart's Law originates in economics. A government struggling with a cobra population offered bounties for cobra skins. Citizens responded rationally — they farmed cobras. Skins handed in went up. The cobra population grew. The metric had been optimised perfectly, and in doing so, became useless.

The metric was designed to represent reality. Once it became the target, it stopped doing that.

This is not a story about bad actors. Everyone behaved logically. The system failed because the incentive structure was misaligned with the actual goal — and optimisation pressure exposed that misalignment completely.

---

## The same failure in ML

In machine learning, this failure mode appears in several forms:

| Term | Description |
|---|---|
| **Goodhart's Law** | A metric collapses when it becomes the target |
| **Specification gaming** | The agent exploits loopholes in how the goal was defined |
| **Reward hacking** | The agent maximises reward in ways the designer did not intend |
| **Wireheading** | The agent manipulates its own reward signal directly |

These are not separate problems. They share one structure: a proxy metric that breaks down under optimisation pressure.

A Snake AI trained to stay alive and score points discovered it could satisfy both conditions by circling endlessly — never progressing, never dying, never winning. The reward was maximised. The goal was abandoned. When researchers applied a Hamiltonian cycle to fix it, the agent beat the game reliably — but it had stopped learning entirely. The metric problem was bypassed, not solved.

---

## The question

Most approaches to this problem attack the reward function — making it more sophisticated, adding constraints, using human feedback to shape it. This project asks a different question:

**What if the problem starts earlier, in what the model is given to learn from?**

A CNN processing raw board state has to discover what matters on its own. It may learn superficial patterns that satisfy the metric without capturing the underlying structure of the problem. But if we feed the model the concepts that actually matter — danger proximity, accessible space, positional ranking — the learning task changes. The model has less room to find unintended shortcuts because the input already encodes intent.

This is the hypothesis. Snake is the controlled environment to test it — simple enough to isolate the variable cleanly, complex enough to produce meaningful failure modes.

---

## Scope

This is an active learning project at the intersection of ML engineering and alignment thinking. The goal is not to solve reward hacking at scale. The goal is to build a clear, reproducible demonstration of how input representation affects the gap between metric and intent — and to develop intuition about where that gap lives and how it can be reduced.

Results, methodology, and open questions are documented in [`RESEARCH.md`](./RESEARCH.md).
The development roadmap is in [`ROADMAP.md`](./ROADMAP.md).

---

## References

- Goodhart, C. (1975). *Problems of Monetary Management: The U.K. Experience*
- Krakovna et al. (2020). *Specification gaming: the flip side of AI ingenuity* — DeepMind
- Amodei et al. (2016). *Concrete Problems in AI Safety* — OpenAI
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* — Chapter 3
