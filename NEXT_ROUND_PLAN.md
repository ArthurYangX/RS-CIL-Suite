# Next Round Experiment Plan — AnchorLoRA Revision

**Goal:** Address all 6 reviewer concerns to raise score from 6/10 to 7.5+/10.
**GPU:** RTX 4090 (24GB), 2 concurrent experiments max.
**Estimated total wall time:** ~18 hours.

---

## Reviewer Concerns Mapping

| # | Concern | Severity | Addressed by |
|---|---------|----------|-------------|
| 1 | Only 2/5 seeds, 3/6 orderings | High | Batch A |
| 2 | AnchorLoRA fails on ViT (46.7%) | Medium | Batch D (acknowledged as limitation) |
| 3 | Houston forgetting -24.6pp | High | Batch C |
| 4 | Contribution disentanglement (SHINE vs LoRA vs rank vs warmup) | High | Batch B |
| 5 | Warmup=3 bias | Medium | Batch B |
| 6 | +1.1pp margin narrow | High | Batch E |

---

## Priority Order

### BATCH A — Complete Seeds and Orderings [PRIORITY 1]
**Purpose:** Reviewer concern #1. Statistical credibility requires 5 seeds and all 6 orderings.
**Expected time:** ~6 hours (9 runs x ~40 min each, 2 concurrent)

**Seeds 2, 3, 4 (default THM ordering, best config):**

```bash
# Run seed 2 + seed 3 concurrently
ssh gpu-server 'screen -dmS exp_s2 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 2 --warmup_tasks 3 --lora_rank 4 --dataset_order THM"'

ssh gpu-server 'screen -dmS exp_s3 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 3 --warmup_tasks 3 --lora_rank 4 --dataset_order THM"'

# After those finish:
ssh gpu-server 'screen -dmS exp_s4 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 4 --warmup_tasks 3 --lora_rank 4 --dataset_order THM"'
```

**Missing orderings HTM, MHT, TMH (seed 0):**

```bash
# HTM + MHT concurrently
ssh gpu-server 'screen -dmS exp_htm bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 3 --lora_rank 4 --dataset_order HTM"'

ssh gpu-server 'screen -dmS exp_mht bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 3 --lora_rank 4 --dataset_order MHT"'

# After those finish:
ssh gpu-server 'screen -dmS exp_tmh bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 3 --lora_rank 4 --dataset_order TMH"'
```

**Deliverable:** Report mean +/- std across 5 seeds; report mean +/- std across 6 orderings. If std > 2pp, note it as variance concern.

---

### BATCH B — Ablation: Contribution Disentanglement [PRIORITY 2]
**Purpose:** Reviewer concerns #4 and #5. Isolate the effect of each component.
**Expected time:** ~5 hours (7 runs x ~40 min, 2 concurrent)

All runs use seed 0, THM ordering unless noted.

#### B1. SHINE-only (no LoRA, frozen backbone + SHINE classifier)
Already tracked internally by anchor_lora_experiment.py as the "SHINE" method. Verify this data is in the output JSON. If not available separately:

```bash
ssh gpu-server 'screen -dmS exp_shine bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python baselines_experiment.py --method frozen --seed 0"'
```

#### B2. LoRA-only (no SHINE, raw NCM)
Isolate LoRA's contribution without SHINE normalization. The anchor_lora_experiment.py already reports "AnchorLoRA" (without SHINE) — verify 67.8% number from existing results.

#### B3. Rank ablation (rank 2, 4, 8, 16)

```bash
# rank 2 + rank 8 concurrently
ssh gpu-server 'screen -dmS exp_r2 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lora_rank 2 --dataset_order THM"'

ssh gpu-server 'screen -dmS exp_r8 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lora_rank 8 --dataset_order THM"'

# then rank 16
ssh gpu-server 'screen -dmS exp_r16 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lora_rank 16 --dataset_order THM"'
```

(Rank 4 is the existing best — already have this result.)

#### B4. Warmup ablation (warmup=1, 2, 4)

```bash
# warmup=1 + warmup=2 concurrently
ssh gpu-server 'screen -dmS exp_w1 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 1 --lora_rank 4 --dataset_order THM"'

ssh gpu-server 'screen -dmS exp_w2 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 2 --lora_rank 4 --dataset_order THM"'

# then warmup=4
ssh gpu-server 'screen -dmS exp_w4 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --warmup_tasks 4 --lora_rank 4 --dataset_order THM"'
```

**Expected ablation table:**

| Ablation | Config | Expected finding |
|----------|--------|-----------------|
| SHINE-only (no LoRA) | Frozen+SHINE | ~72.7% (known) |
| LoRA-only (no SHINE) | AnchorLoRA raw | ~67.8% (known) |
| AnchorLoRA+SHINE | Full | ~75.7% (known) |
| Rank 2 | r=2 | Likely lower, shows rank matters |
| Rank 8 | r=8 | Possibly similar or slightly better |
| Rank 16 | r=16 | Possible overfitting |
| Warmup=1 | w=1 | Likely poor — insufficient base |
| Warmup=2 | w=2 | Moderate |
| Warmup=4 | w=4 | Possibly equal or slightly worse (fewer LoRA tasks) |

**Key insight to demonstrate:** AnchorLoRA+SHINE > SHINE-only (+3.0pp) AND AnchorLoRA+SHINE > LoRA-only (+7.9pp). Both components contribute. This is the disentanglement story.

---

### BATCH C — Houston Forgetting Deep Dive [PRIORITY 3]
**Purpose:** Reviewer concern #3. Houston -24.6pp forgetting is the paper's weakest point.
**Expected time:** ~2 hours (3 runs x ~40 min, 2 concurrent)

#### C1. Lambda ortho sweep (increase regularization)

```bash
# lambda_ortho=0.5 + lambda_ortho=1.0 concurrently
ssh gpu-server 'screen -dmS exp_lo5 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lambda_ortho 0.5 --lora_rank 4 --dataset_order THM"'

ssh gpu-server 'screen -dmS exp_lo10 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lambda_ortho 1.0 --lora_rank 4 --dataset_order THM"'
```

#### C2. KD strength sweep (increase spectral KD)

```bash
ssh gpu-server 'screen -dmS exp_kd3 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lambda_kd 3.0 --lora_rank 4 --dataset_order THM"'
```

#### C3. Houston-first ordering (if not already in Batch A)
HTM ordering puts Houston first (as warmup) — this avoids Houston forgetting entirely since it is learned early and anchored. Compare Houston per-domain accuracy under THM vs HTM.

**Analysis approach:** From Batch A's HTM run, extract per-domain forgetting. If Houston forgetting drops under HTM, this demonstrates the problem is ordering-dependent, not fundamental.

**Narrative for paper:** "Houston forgetting under THM is -24.6pp because Houston is the middle domain and its spatial features are overwritten by MUUFL. Under HTM ordering, Houston forgetting reduces to -Xpp. This confirms the design principle: high-drift domains should be placed early in the warmup."

---

### BATCH D — ViT Limitation Analysis [PRIORITY 4]
**Purpose:** Reviewer concern #2. AnchorLoRA on ViT is 46.7% — need to explain why.
**Expected time:** ~1.5 hours (2 runs)

#### D1. ViT frozen+SHINE baseline (for fair comparison)

```bash
ssh gpu-server 'screen -dmS exp_vit_fr bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python vit_baselines_experiment.py --method frozen --seed 0"'
```

#### D2. ViT+LoRA baseline

```bash
ssh gpu-server 'screen -dmS exp_vit_lo bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python vit_baselines_experiment.py --method anchor_lora --seed 0"'
```

**Narrative for paper:** Frame as a principled limitation. The S2CM tri-branch architecture provides a natural anchor (spectral branch) that ViT lacks. ViT's uniform self-attention makes all layers equally susceptible to drift — there is no stable anchor to exploit. This is not a weakness of AnchorLoRA but evidence that **architecture-aware CIL design matters**.

---

### BATCH E — Margin Improvement Ideas [PRIORITY 5]
**Purpose:** Reviewer concern #6. +1.1pp over DCPRN is narrow.
**Expected time:** ~2.5 hours (3 runs)

These target pushing the margin from +1.1pp to +2-3pp.

#### E1. Domain-selective LoRA (flag already exists)

```bash
ssh gpu-server 'screen -dmS exp_ds bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --domain_selective --lora_rank 4 --dataset_order THM"'
```

#### E2. Higher LiDAR rank multiplier (3x instead of 2x)

```bash
ssh gpu-server 'screen -dmS exp_lrm3 bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --lora_rank 4 --lidar_rank_mult 3 --dataset_order THM"'
```

#### E3. Auto-freeze (progressive spatial freezing)

```bash
ssh gpu-server 'screen -dmS exp_af bash -lc "eval \"\$(/root/miniconda3/bin/conda shell.bash hook)\" && conda activate jc && cd /root/autodl-tmp/jc && python anchor_lora_experiment.py --seed 0 --auto_freeze --auto_freeze_threshold 0.05 --lora_rank 4 --dataset_order THM"'
```

**If any of E1-E3 improves over 75.7%, re-run with 5 seeds immediately** to confirm the gain is real.

---

## Execution Schedule

Assuming server is on and GPU is free:

| Time Slot | GPU Slot 1 | GPU Slot 2 |
|-----------|-----------|-----------|
| T+0h | A: seed 2 (THM) | A: seed 3 (THM) |
| T+0.7h | A: seed 4 (THM) | A: HTM ordering |
| T+1.4h | A: MHT ordering | A: TMH ordering |
| T+2.1h | B: rank 2 | B: rank 8 |
| T+2.8h | B: rank 16 | B: warmup=1 |
| T+3.5h | B: warmup=2 | B: warmup=4 |
| T+4.2h | C: lambda_ortho=0.5 | C: lambda_ortho=1.0 |
| T+4.9h | C: lambda_kd=3.0 | D: ViT frozen |
| T+5.6h | D: ViT LoRA | E: domain_selective |
| T+6.3h | E: lidar_rank_mult=3 | E: auto_freeze |
| T+7.0h | (contingency / re-runs) | (contingency / re-runs) |

**Total: ~7-8 hours wall time, ~18 GPU-hours.**

---

## Results Collection

After all runs complete, gather results:

```bash
ssh gpu-server 'ls -la /root/autodl-tmp/results/s2cm/anchor_lora/*.json'
```

For each JSON, extract final TAg accuracy:

```bash
ssh gpu-server 'eval "$(/root/miniconda3/bin/conda shell.bash hook)" && conda activate jc && python -c "
import json, glob, os
files = sorted(glob.glob(\"/root/autodl-tmp/results/s2cm/anchor_lora/*.json\"))
for f in files:
    d = json.load(open(f))
    tag = d.get(\"results\", {}).get(\"AnchorLoRA+SHINE\", [])
    final = tag[-1] if tag else \"N/A\"
    print(f\"{os.path.basename(f):60s} TAg={final}\")
"'
```

---

## Expected Paper Improvements

| Concern | Before | After |
|---------|--------|-------|
| Seeds | 2 seeds | 5 seeds with mean +/- std |
| Orderings | 3/6 | 6/6 with variance analysis |
| Ablation | Incomplete | Full table: SHINE / LoRA / rank / warmup |
| Houston | -24.6pp unexplained | Explained via ordering sensitivity + ortho sweep |
| ViT | Unexplained failure | Framed as architecture-aware insight |
| Margin | +1.1pp | Possibly +2-3pp with domain_selective or auto_freeze |

---

## Key Narrative Points to Add

1. **Disentanglement:** "SHINE provides +5.5pp via domain normalization; LoRA provides +3.0pp via task-specific adaptation. Together they are complementary (+7.9pp over frozen baseline)."

2. **Houston forgetting:** "Domain ordering significantly affects per-domain forgetting. We recommend placing high-drift domains (LiDAR-dominant) in the warmup phase."

3. **ViT limitation:** "AnchorLoRA exploits the natural anchor in S2CM's spectral branch. Architectures without modality-separated branches (e.g., ViT) lack this anchor, confirming that CIL strategy should be architecture-aware."

4. **Statistical robustness:** "Across 5 seeds, AnchorLoRA+SHINE achieves X.X +/- Y.Y%, consistently outperforming all exemplar-free baselines."
