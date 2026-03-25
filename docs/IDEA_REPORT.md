# Idea Discovery Report (v2 — Data-Driven)

**Direction**: Multimodal (HSI+LiDAR) Exemplar-Free Class-Incremental Learning
**Date**: 2026-03-22
**Pipeline**: research-lit → idea-creator (GPT-5.4 xhigh) → novelty-check → research-review (GPT-5.4 xhigh)

## Executive Summary

After comprehensive literature survey, GPT-5.4 brainstorming (10 ideas), deep novelty verification on top 3, and brutal critical review, we converge on **BRACE: Branch-wise Re-centering and Ambiguity Capsule Experts** — a synthesis of three surviving ideas (SHINE + BRIO + COPE). The key insight from our marathon experiment is that **cross-domain feature drift under a frozen encoder** is the real bottleneck (Houston drops to 42.2%), not classical prototype drift. A critical gate experiment (SHINE = per-domain whitening + cosine NCM) is running to determine paper scope.

**Gate decision:** If SHINE solves Houston → finding paper (TGRS). If not → full BRACE method (TIP).

---

## Literature Landscape

### Key State-of-the-Art (2023-2026)

| Paper | Venue | Core Idea | Relevance |
|-------|-------|-----------|-----------|
| LDC | ECCV 2024 | Learnable drift compensation via projector | Closest to CATCH; assumes moving backbone |
| DPCR | ICML 2025 | Dual-projection for semantic shift | Linear transforms on stored stats |
| SLDC | arXiv 2025 | Linear transition operators for drift | Chain-of-projectors approach |
| DATS | arXiv 2025 | Distance-aware temperature scaling for CIL | Closest to DART |
| T-CIL | CVPR 2025 | Adversarial temp scaling for CIL | Crowded temperature space |
| FeCAM | NeurIPS 2023 | Per-class Mahalanobis + Tukey + shrinkage | Frozen backbone + covariance |
| DSRSD-Net | arXiv 2025 | Shared/residual multimodal decomposition | Close to SPARC concept |
| ACPM | CVPR 2025 | Adapter merging + centroid prototype mapping | Frozen backbone CIL |
| NAPA-VQ | ICCV 2023 | Neighborhood-aware prototype augmentation | VQ-based generation |
| MPM | CVPR 2025 | Multi-prototype per domain | Cross-domain CIL |
| PINA | ECCV 2024 | Domain-specific alignment for domain-incremental | Non-exemplar |
| PMPT | TIP 2026 | Frozen backbone + prompt tuning + NCM | Direct competitor |

### Structural Gaps
1. **No cross-dataset marathon CIL for multimodal RS** — our strongest novelty
2. **No domain equalization methods for frozen-backbone CIL** — static encoder, cross-domain overlap
3. **No branch-level scoring calibration** — all methods treat features as single vector
4. **No targeted cross-domain ambiguity resolution** — sparse pairwise experts

---

## Ranked Ideas (10 generated, 4 survive)

### Eliminated Ideas

| Idea | Score | Kill Phase | Kill Reason |
|------|-------|------------|-------------|
| CATCH | 5.5/10 | Novelty (Phase 3) | Too close to LDC/DPCR/SLDC lineage; crowded drift compensation space |
| DART | 6/10 | Review (Phase 4) | Crowded temperature scaling space (DATS, T-CIL); not the real bottleneck |
| HOTA | — | Review (Phase 4) | OT is overkill; hard to stabilize without semantic correspondences |
| TIDE | — | Review (Phase 4) | Test-time adaptation muddies protocol purity in marathon setting |
| CALM | — | Review (Phase 4) | Reads as prototype-repositioning variant; COPE is cleaner |
| SPARC | 6/10 | Review (Phase 4) | Trainable regressors weaken frozen-backbone appeal; DSRSD-Net overlap |

### Surviving Ideas

| Idea | Role | Score |
|------|------|-------|
| **SHINE** | Core baseline / first module | 7/10 as baseline |
| **BRIO** | Support module (query-wise branch reliability) | 7/10 |
| **COPE** | Main novelty (sparse pairwise conflict experts) | 8/10 |
| **VAPR** | Reduced form (capsule sampling for local experts) | 7/10 |

### 🏆 Recommended Synthesis: BRACE (Branch-wise Re-centering and Ambiguity Capsule Experts)

**Pipeline:**
1. **Domain canonicalization (SHINE):** Per-domain/branch diagonal whitening with count-shrunk identity regularization → shared canonical space
2. **Adaptive branch weighting (BRIO):** Top-2 margin + class variance + cross-branch agreement → query-specific branch weights (replaces Fisher)
3. **Ambiguity experts (COPE+VAPR):** Mine high-conflict cross-domain class pairs in canonical space. For those pairs only, sample pseudo-features from stored capsules and fit tiny pairwise classifiers. Invoke only when global top-2 margin is small.

**Method novelty:** 7.0-7.5/10
**With marathon setting + analysis:** 7.8-8.2/10

**Differentiators:**
- NOT LDC/DPCR: those assume moving backbone; ours is frozen, addressing static cross-domain overlap
- NOT T-CIL/DATS: changes geometry and resolves ambiguity, not just temperatures
- NOT PMPT: no prompt tuning, no session-specific embedding adaptation

---

## Critical Gate Experiment (RUNNING)

**SHINE gate:** Per-domain whitening + cosine NCM on marathon data.

- If Houston improves by >+8% → paper = finding + simple method → **TGRS**
- If Houston improves by +3% to +8% → SHINE as base, add BRACE → **TIP**
- If Houston improves by <+3% → need full BRACE → **TIP**

---

## Key Experimental Evidence

### Marathon CIL (Trento→Houston→MUUFL, 9 tasks, 32 classes)
| Task | Dataset | #Seen | Avg TAg | Per-dataset at final |
|------|---------|-------|---------|---------------------|
| 0 | Trento | 2 | 99.9% | Trento: 86.3% |
| 3 | Houston | 11 | 85.2% | Houston: 42.2% |
| 8 | MUUFL | 32 | 64.4% | MUUFL: 64.8% |

**Core finding:** Cross-domain feature drift (not classifier bias) is the real bottleneck. The frozen backbone's features are simply not discriminative enough for unseen domains.

---

## Next Steps

- [ ] SHINE gate experiment result → determines paper scope
- [ ] If BRACE needed: implement COPE pairwise experts
- [ ] Multi-seed validation (seeds 0, 42, 123)
- [ ] CIL baselines comparison (LwF, EWC, fine-tune)
- [ ] /auto-review-loop to iterate until submission-ready
