# Roadmap

Project direction and open questions. Updated as work progresses.

---

## Phase 1 — Reproduce and confirm

- [ ] Implement standard Snake environment
- [ ] Train CNN agent on raw board state
- [ ] Document the failure mode: survival without progression
- [ ] Visualise learned behaviour patterns

---

## Phase 2 — Redesign input representation

- [ ] Define structured feature set (danger map, tail proximity, square ranking)
- [ ] Retrain agent on structured input with same reward function
- [ ] Compare behaviour and performance against Phase 1 baseline
- [ ] Document whether the failure mode changes, reduces, or shifts

---

## Phase 3 — Scoring function exploration

- [ ] Test alternative reward functions on both input types
- [ ] Explore whether structured input allows simpler, more robust reward design
- [ ] Investigate whether reward shaping can be partially learned from demonstrations

---

## Phase 4 — Generalisation (stretch goal)

- [ ] Apply same input-shaping logic to a second environment
- [ ] Test whether Phase 2 findings hold outside Snake
- [ ] Write up findings as a short technical post

---

## Open questions

- Does structured input reduce reward hacking, or does it just move it elsewhere?
- Is there a principled way to decide which features belong in the input layer?
- At what point does feature engineering become a form of specification gaming itself?
