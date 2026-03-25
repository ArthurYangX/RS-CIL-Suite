# Research Idea Report v2: Breaking the Frozen-Backbone Representation Ceiling

**Direction**: Representation-level methods for multimodal HSI+LiDAR exemplar-free cross-domain CIL
**Generated**: 2026-03-23
**Pipeline**: WebSearch literature → GPT-5.4 xhigh brainstorm → novelty check → GPT-5.4 xhigh devil's advocate
**Ideas evaluated**: 10 generated → 4 survived filtering → 2 recommended for implementation

---

## Executive Summary

**Top recommendation: SART + PROST unified framework.**

Use the stable spectral branch as a long-term semantic anchor, and learn lightweight transport modules that correct drifting spatial features (SART) and prototypes (PROST) back to a canonical geometry — without touching the backbone. This is physically motivated by the measured HSI/LiDAR drift asymmetry (2×), differentiated from all adapter/LoRA/prompt CIL methods, and directly addresses the proven representation bottleneck.

**Implementation order**: PROST first (cheapest falsification test, ~1 day), then SART (main method, ~2-3 days).

---

## Literature Landscape

### Adapter/LoRA-Based CIL (2024-2025)
| Paper | Venue | Core Idea | Limitation for Us |
|-------|-------|-----------|-------------------|
| CL-LoRA | CVPR 2025 | Dual adapter (shared + task-specific) | Single modality, no cross-domain |
| InfLoRA | CVPR 2024 | Null-space constrained LoRA | Single modality, weight-space |
| LoRA- | CVPR 2025 | LoRA subtraction for drift-resistant space | Single modality |
| EASE | CVPR 2024 | Per-task expandable adapter subspaces | Single modality, no sensor shift |
| RanPAC | NeurIPS 2023 | Random projection + frozen prototype | No cross-domain, no multimodal |

### Multimodal CIL (2024-2025)
| Paper | Venue | Core Idea | Limitation for Us |
|-------|-------|-----------|-------------------|
| Cross-Modality Adapters | arXiv 2511 | MoE cross-modal adapters + repr alignment | Replay-based, not RS-specific |
| ATLAS | arXiv 2410 | Two-stage adapter knowledge reuse | Upstream/downstream, not CIL |
| DisCo | AAAI 2025 | Domain shift as forgetting alleviator | Natural images, symmetric drift |
| PMPT | TIP 2026 | Prompt tuning for HSI CIL | Our direct competitor |

### Critical Gap
**ALL** adapter/LoRA CIL methods target single-modality ViT on natural images. **NONE** address:
1. Multi-branch architecture with modality-specific drift rates
2. Cross-domain CIL with genuine sensor differences (different HSI bands, LiDAR channels)
3. The physical asymmetry where spectral features are stable but spatial features drift 2×

---

## Recommended Ideas (Ranked)

### 🏆 Idea 1: SART — Spectral-Anchored Residual Transport

- **Hypothesis**: The frozen spectral branch provides a stable semantic reference frame; lightweight sample-wise transport modules can correct drifting spatial features back to a canonical space using spectral conditioning.
- **Core mechanism**:
  ```
  z_s (spectral, 64d) → FROZEN (anchor)
  z_h (HSI-spatial, 64d) → z_h' = z_h + T_h(z_h, z_s, d_t)
  z_l (LiDAR-spatial, 64d) → z_l' = z_l + T_l(z_l, z_s, d_t)

  Classify on [z_s, z_h', z_l'] with branch cosine NCM

  T_h, T_l: FiLM-style 2-layer residual MLPs (64+64+16 → 64)
  d_t: domain signature from batch feature statistics
  ```
- **Training losses**:
  - CE / SupCon on current task data
  - Old-prototype consistency: transport of old prototypes should preserve their relative geometry
  - Cross-modal agreement: transported spatial features should align with spectral semantics
- **Why it might work**:
  - Directly leverages spectral stability (Fisher 50-80%) as anchor
  - SHINE proves first-order transport helps (+24%); SART is the nonlinear, sample-conditioned extension
  - Only modifies spatial branches (the ones that actually drift)
- **Why it might fail**: With only 2-5 classes/task and 3 domains, transport modules may memorize domain ID instead of learning reusable correction
- **Novelty**: 9/10 — No prior work does spectral-anchored, branch-asymmetric, sample-level feature transport in multimodal CIL
- **Closest prior work**: CL-LoRA / InfLoRA (weight-space LoRA, single modality) — SART is feature-space, cross-modal, physically anchored
- **Feasibility**: ~60K-100K params, trainable on 4090 in minutes/task
- **Scores**: Effectiveness 9, Novelty 9, Simplicity 7, Safety 8, Differentiation 9, Feasibility 9 → **Avg 8.5**
- **Reviewer's likely objection**: "This is just a conditional feature adapter" → Defense: the spectral anchoring and branch asymmetry are physically motivated, not generic
- **Minimum viable experiment**: LiDAR-only transport on Trento→Houston. If Houston rises over SHINE without Trento drop → proceed.

### 🥈 Idea 2: PROST — Prototype Reconstruction from Spectral Transport

- **Hypothesis**: Cross-domain confusion is largely a prototype misplacement problem; spectral prototypes can predict how spatial prototypes should shift across domains.
- **Core mechanism**:
  ```
  For each class c, store prototype triplet (p_s^c, p_h^c, p_l^c)

  Learn correctors: D_h(p_s, d_t) → Δp_h, D_l(p_s, d_t) → Δp_l

  Corrected prototypes: (p_s^c, p_h^c + Δp_h, p_l^c + Δp_l)
  ```
- **Training**: Only current-task prototypes + angle-preservation loss over old prototype graph
- **Why it might work**: Cheapest test of the spectral-anchor hypothesis. If prototype-level correction alone helps, sample-level (SART) is justified.
- **Why it might fail**: Within-class scatter under sensor change may dominate over mean shift
- **Novelty**: 8/10 — Not deformable prototypes (class-specific); this is class-agnostic spectral-conditioned transport
- **Scores**: Effectiveness 8, Novelty 8, Simplicity 9, Safety 9, Differentiation 8, Feasibility 10 → **Avg 8.7**
- **Minimum viable experiment**: Learn corrector on Houston new classes, apply to Trento old prototypes, check margin improvement.

---

## Combined Framework: SART + PROST

**Two-scale spectral-anchored transport:**
1. **Global (PROST)**: Transport prototypes to fix cross-domain class geometry
2. **Local (SART)**: Transport individual samples to fix instance-wise domain distortion

**Paper narrative for IEEE TIP:**
- **Core claim**: Frozen multimodal CIL failure under cross-domain sensor shift is **branch-asymmetric representation drift**, not generic forgetting
- **Empirical premise**: Spectral stable + dominant; spatial drifts 2×; SHINE = first-order ceiling; fine-tuning = uncontrollable drift
- **Method principle**: Spectral branch as semantic anchor → transport spatial features/prototypes to canonical geometry
- **Technical contribution**: SART (sample transport) + PROST (prototype transport), unified by spectral anchoring
- **Differentiation**: Not prompt tuning (vs PMPT), not backbone LoRA (vs CL-LoRA), not expandable adapters (vs EASE); physically motivated for multimodal RS

---

## Backup Ideas (Not Recommended for v1)

### Idea 3: AIMS — Asymmetric Invariant Modality Splitting
- Split spatial branches into invariant (aligned with spectral) + nuisance components
- Classify only on invariant codes
- **Highest conceptual novelty** (9/10) but **highest risk**: factorization may collapse with only 3 domains
- Save as high-risk backup if SART underperforms

### Idea 4: SEER — Sparse Expandable Error Residuals
- Atom bank of reusable distortion primitives for spatial branches
- **Deprioritized**: too much overlap with EASE/MoE adapter papers after novelty check

### Idea 5: CABIN — Counterfactual Asymmetric Branch Interpolation
- Synthetic hard negatives by branch swapping to attack TAw/TAg gap
- **Good as a regularizer/booster** on top of SART, not standalone

### Idea 6: ARGO — Anchor-Based Representation Generative Replay
- Use spectral prototypes to generate pseudo spatial features for old classes
- Interesting but generator fidelity is a risk with cross-sensor data

---

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| SCOUT (Spectral-Conditioned Optimal Prototype Transport) | Too few classes for reliable transport map early in stream |
| EDGE (Error-Driven Graph Editing) | Too heuristic, modest gains expected |
| GPF (Geodesic Prototype Flow) | Normalizing flow is overkill and unstable for 2-5 cls/task |
| RISE (Replay-Free Invariant Spectral Expansion) | Too similar to PCA-style compression (PCA already tested, +2% max) |

---

## Positioning Against Competition

| Competitor | Their Route | Our Differentiation |
|------------|-------------|---------------------|
| PMPT (TIP 2026) | Prompt tuning + NCM | Feature-space transport, no prompts, branch-asymmetric |
| CL-LoRA (CVPR 2025) | Weight-space dual LoRA | Feature-space, cross-modal, physically anchored |
| InfLoRA (CVPR 2024) | Null-space LoRA | Not weight-space; spectral anchor is physical, not null-space |
| EASE (CVPR 2024) | Expandable adapter subspaces | Not expanding; transporting to canonical space |
| Cross-Modal Adapters (arXiv 2511) | MoE cross-modal + replay | Exemplar-free, RS-specific sensor physics |
| DisCo (AAAI 2025) | Domain shift helps separation | Our problem is asymmetric sensor drift, not task separation |

---

## Implementation Plan

### Step 1: PROST (1 day, cheapest falsification)
- Implement prototype corrector D_h, D_l
- Test on Trento→Houston transition
- **Go/No-Go**: If Houston prototype margins improve → proceed to SART

### Step 2: SART (2-3 days)
- Implement sample-level transport T_h, T_l
- Train with CE + old-prototype consistency + cross-modal agreement
- Test on full marathon

### Step 3: Combine SART + PROST (1 day)
- PROST as prototype regularizer for SART
- Ablation: SART alone vs PROST alone vs SART+PROST

### Step 4: Full evaluation (2-3 days)
- Multi-seed (3 seeds)
- Single-dataset + marathon
- Ablation table
- Compare with SHINE, iCaRL, Replay-Herding

**Total estimated time: ~1 week**

---

## Success Targets

| Setting | SHINE (current ceiling) | Target (SART+PROST) |
|---------|------------------------|---------------------|
| Marathon TAg | 72.8% | ≥ 78% |
| Marathon Houston | 68.6% | ≥ 75% |
| Marathon Trento | 87.0% | ≥ 87% (no degradation) |
| Marathon MUUFL | 70.7% | ≥ 72% |
| Trento single | 93.9% | ≥ 93% |
| Houston single | 69.4% | ≥ 75% |

---

## GPT-5.4 Thread
- Thread ID: `019d1665-25ff-73e2-9e75-1ddd2dc79dd6`
- Rounds: 2 (brainstorm + devil's advocate)
- Model: gpt-5.4 xhigh reasoning
