# Research Review: SART + PROST Method Proposal

**Date**: 2026-03-23
**Reviewer**: GPT-5.4 xhigh via Codex MCP
**Thread ID**: `019d1671-cb59-7ee3-a18c-a48b486d898a`
**Rounds**: 2

---

## Round 1: Initial Assessment

### Score: 4/10 (current state, without SART implemented)

**Conditional**: If SART works + proper controls → could reach 7/10 (borderline TIP / weak accept)

### Critical Weaknesses (Ranked)

1. **Main breakthrough doesn't exist yet** — SART/PROST is proposed but not implemented
2. **Benchmark confounded by initialization and order** — Trento-first structurally privileges home domain
3. **40pp gap mixes multiple factors** — domain shift + class count + dataset difficulty + sensor differences
4. **"Ceiling" overclaiming** — 10 failed methods ≠ proven ceiling, just a strong baseline
5. **LiDAR drift measured on wrong backbone** — TransErr data from earlier backbone, not S²CM
6. **Spectral anchor not yet defensible** — different band counts (63/144/72) challenge "invariance" claim
7. **Domain ID may be privileged information** — per-domain whitening needs explicit test-time protocol
8. **SART/PROST risks looking like engineering cocktail** — multiple known ideas stacked
9. **Single custom backbone** — may be diagnosing S²CM pathology, not general phenomenon
10. **"Benchmark" overclaim** — need full public release to call it a benchmark

### Narrative Assessment

**Coherent version (recommended)**:
Marathon protocol → SHINE diagnosis → SART method (one thesis)

**Incoherent version (current drift)**:
Benchmark + normalization + negative results + transport method (four stitched papers)

**Advice**: SHINE and negative evidence should SUPPORT the diagnosis, not compete with SART as equal contributions.

---

## Round 2: Rebuttal Action Plan

### Top 5 Actions to Move from 4/10 to 7/10

1. **Implement minimal SART and prove it on hardest transition**
   - SART-only, no PROST, frozen spectral, residual adapters on spatial branches
   - If no gain → stop

2. **Repair protocol validity**
   - Run Trento-first, Houston-first, MUUFL-first orderings
   - Frame Trento-home as "home-to-away stress test" (design, not confound)

3. **Re-measure branch drift on actual S²CM backbone**
   - Before/after SHINE, before/after SART
   - Verify LiDAR > HSI drift asymmetry on current model

4. **Benchmark against proper baselines + oracle-normalized scores**
   - Compare to PMPT, EASE, InfLoRA, CL-LoRA (or explain why not feasible)
   - Report results as fraction of joint-training oracle per domain

5. **Cut paper to one thesis**
   - Main: Marathon → SHINE diagnosis → SART method
   - Move 10+ negative methods to supplement (keep 2-3 most informative)

### Minimum Viable SART Experiment

**Setting**: Single domain jump (Trento tasks → first Houston task only)

**Compare**: Frozen-NCM, SHINE, SART, SART-without-anchor

**Success bar**:
- +3pp over SHINE on TAg after Trento→Houston jump
- ≤1pp Trento degradation vs SHINE
- Visible LiDAR drift reduction
- Anchored version beats non-anchored

### Results-to-Claims Matrix

| SART Δ vs SHINE | Allowed Claim | TIP Viability |
|-----------------|---------------|---------------|
| < +2pp | "Competitive extension" | NOT enough for TIP method paper |
| +2 to +4pp | "Consistent improvement" | Weak, needs very strong controls |
| **+4 to +6pp** | **"Residual bottleneck can be reduced by anchored transport"** | **First viable TIP range** |
| +6 to +8pp | "Dominant remaining error is mitigable" | Strong method story |
| > +8pp | "Major advance" | Breakthrough if survives order rotation |

**All thresholds require**: TAg metric, not just TAw; balanced per-domain gains; flat forgetting; multi-seed; ≥2 start orders.

### What to Cut

1. ~~"Ceiling"~~ → "strong post-hoc baseline"
2. Most of 10+ failed methods → supplement (keep 2-3 in main text)
3. PROST → only if adds ≥1.5-2pp over SART alone
4. ~~"Spectral invariance"~~ → "empirically more stable"
5. Unknown-domain routing → only if tested, otherwise state known-domain
6. ~~"Benchmark"~~ → "protocol" unless full public release

### Paper Structure (Assuming SART +5pp)

| Section | Content | Key Table/Figure |
|---------|---------|------------------|
| 1. Introduction | One thesis: cross-domain CIL fails → SHINE diagnosis → SART fixes residual gap | Fig 1: teaser |
| 2. Protocol | Marathon CIL definition, metrics, storage, test-time assumptions | Tab 1: datasets |
| 3. Diagnosis | Single vs marathon gap, SHINE, branch drift analysis | Tab 2: baselines, Fig 2: trajectories, Fig 3: drift |
| 4. Method | SART architecture + losses (PROST only if earned) | Fig 4: architecture |
| 5. Experiments | Main results, start-order rotation, oracle-normalized | Tab 3: main, Tab 4: normalized |
| 6. Analysis | Ablation, mechanism, few-shot | Tab 5: ablation, Fig 5: per-domain |
| 7. Limitations | Domain ID, single backbone, spectral stability is empirical | — |

### Implementation Priority

```
1. SART-only → test on Trento→Houston jump
2. SART ablations (anchor vs no-anchor, symmetric vs asymmetric)
3. Full marathon SART
4. Rotated start orders (Houston-first, MUUFL-first)
5. PROST → only if SART already works
```

---

## Key Takeaways

1. **The paper is NOT ready without SART working.** Current state = diagnostic study, not method paper.
2. **SART is the make-or-break.** If +4pp over SHINE with clean controls → TIP viable.
3. **Keep it simple.** One method (SART), one thesis, one strong baseline (SHINE).
4. **Protocol rotation is essential.** At least 3 start orders to defend the benchmark claim.
5. **Re-measure drift on current backbone.** The asymmetry story needs S²CM-specific evidence.
6. **PROST is on probation.** Don't include unless it clearly adds value over SART alone.
