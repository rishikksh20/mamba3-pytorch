# Mamba-3: Clean Reference Implementation

A clean, readable, from-scratch PyTorch implementation of **Mamba-3** — a selective state space model that addresses the three core limitations of Mamba-2. No Triton/CUDA kernels or TileLang optimizations; designed for understanding the algorithm.

**Paper:** [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)  
**Authors:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

---

## Core Ideas

Mamba-3 introduces three improvements over Mamba-2:

### 1. Exponential-Trapezoidal Discretization

Mamba-2 used Zero-Order Hold (exponential-Euler), a first-order approximation that loses accuracy at large step sizes. Mamba-3 adopts the **trapezoidal rule**: it averages the `B*x` contribution at time `t-1` and time `t` before applying the state decay.

```
h_t = exp(A·dt_t) · h_{t-1}  +  dt_t · trap_t · (B_t·x_t + B_{t-1}·x_{t-1}) / 2
```

`trap_t` is a learned sigmoid gate that continuously blends between pure Euler (`trap=0`) and full trapezoidal averaging (`trap=1`).

### 2. Complex-Valued (Rotary) State Space

Real-valued SSM hidden states cannot easily represent oscillatory or rotational patterns. Mamba-3 applies **Rotary Position Embeddings (RoPE)** to the B and C (key/query) projections, giving the state an effective complex-valued structure.

A small "angle" projection is learned per head and accumulated over time scaled by `dt`, then used to rotate B and C before each SSM update. This lets the model track phase-dependent and positional dependencies without an explicit complex number type.

### 3. Multi-Input Multi-Output (MIMO) Formulation

Mamba-2 is **SISO**: one input vector drives one output via one SSM state (outer-product update of shape `P×D`). During autoregressive decoding the GPU is memory-bandwidth bound, so this is inefficient.

**MIMO** reuses the *same* D-dimensional SSM state for `R` rank streams simultaneously:

| | SISO | MIMO |
|---|---|---|
| State shape | `(H, P, D)` | `(H, D)` |
| State update | outer product `x ⊗ B` → P×D write | sum of R rank-1 scalar·D-vec terms |
| Output | C @ h → P | R scalars up-projected via `mimo_o` |
| FLOPs/byte ratio | low (decode bound) | R× higher |

B and C projections also get rank-R counterparts (K and Q in the attention analogy).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `L` | Sequence length |
| `H` | Number of SSM heads (`d_inner / headdim`) |
| `P` | Headdim — per-head feature dimension |
| `D` | `d_state` — SSM state size per head |
| `R` | `mimo_rank` — number of MIMO streams (R=1 for SISO) |
| `G` | `ngroups` — B/C projections shared across G heads |

---

## File Structure

```
mamba3.py          — full self-contained implementation (this is all you need)
orig_ref_code.py   — original reference code for comparison
research_notes.md  — notes taken while studying the paper
```

### What's in `mamba3.py`

| Component | Description |
|-----------|-------------|
| `RMSNorm` | Standard RMS layer normalization (used for B/C projections) |
| `build_rope_freqs` | Standard RoPE inverse-frequency schedule |
| `apply_rope` | Rotates pairs of dimensions by given angles |
| `mamba3_siso_scan` | Sequential SSM scan for SISO mode — clear loop over timesteps |
| `mamba3_mimo_scan` | Sequential SSM scan for MIMO mode — shared D-dim state, R rank streams |
| `Mamba3` | Main module: input projection → SSM scan → output projection |
| `Mamba3.step` | Single autoregressive decode step (updates states in-place) |
| `Mamba3.allocate_inference_cache` | Allocates zero states for decoding |
| `MambaBlock` | Residual block: RMSNorm → Mamba3 → add |
| `MambaLMHeadModel` | Full stacked language model with embedding + LM head |
| `MambaConfig` | Dataclass for model hyperparameters |
| `count_parameters` | Utility to count trainable/total parameters |

---

## Usage

### SISO (standard mode)

```python
import torch
from mamba3 import Mamba3

model = Mamba3(
    d_model=256,
    d_state=64,
    expand=2,
    headdim=32,
    ngroups=1,
    is_mimo=False,
)

x = torch.randn(2, 128, 256)   # (batch, seq_len, d_model)
y = model(x)                   # (2, 128, 256)
```

### MIMO mode

```python
model = Mamba3(
    d_model=256,
    d_state=64,
    expand=2,
    headdim=32,
    ngroups=1,
    is_mimo=True,
    mimo_rank=4,   # R parallel streams
)

y = model(x)  # same input/output shape
```

### Autoregressive decode (one token at a time)

```python
# Allocate states — shapes differ between SISO and MIMO
angle_state, ssm_state, bx_prev = model.allocate_inference_cache(batch_size=2)

# SISO  ssm_state: (B, H, P, D)
# MIMO  ssm_state: (B, H, D)   ← no P dimension (projected away by mimo_x)

u = torch.randn(2, 256)   # single token
out, angle_state, ssm_state, bx_prev = model.step(u, angle_state, ssm_state, bx_prev)
```

### Full language model

```python
from mamba3 import MambaLMHeadModel, MambaConfig

cfg = MambaConfig(
    d_model=2048,
    n_layer=24,
    vocab_size=50277,
    ssm_cfg={"is_mimo": True, "mimo_rank": 4},
)
model = MambaLMHeadModel(cfg)

input_ids = torch.randint(0, 50277, (1, 512))
logits = model(input_ids)   # (1, 512, vocab_size)
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | — | Token embedding dimension |
| `d_state` | 128 | SSM state size per head (D) |
| `expand` | 2 | Inner dim multiplier; `d_inner = expand * d_model` |
| `headdim` | 64 | Features per SSM head (P); `nheads = d_inner / headdim` |
| `ngroups` | 1 | Groups for B/C projection sharing |
| `rope_fraction` | 0.5 | Fraction of state dims that rotate (0.5 or 1.0) |
| `dt_min/max` | 0.001/0.1 | Range for initial time-step values |
| `is_mimo` | False | Enable MIMO formulation |
| `mimo_rank` | 4 | Number of parallel MIMO streams R |

---

## Implementation Notes

- **No custom kernels** — everything is plain PyTorch with `einops`. The sequential scan loops are correct but `O(L)` serial; production code uses parallel chunk scans.
- **SISO vs MIMO state shape** is the most important distinction:
  - SISO state: `(B, H, P, D)` — headdim × d_state outer product
  - MIMO state: `(B, H, D)` — x is first projected to R scalars via `mimo_x`, then summed into a shared D-dim state; P is completely projected away
- **B/C bias** (`B_bias`, `C_bias`, initialized to 1) is added *after* RMS norm and *after* group→head expansion, with shape rearranged to `(R, H, D)` for correct broadcasting.
- **Per-head RoPE** — angle increments are `dt_h * angle_raw` independently per head, not a single mean-dt scalar shared across heads.
- **Trapezoidal memory** (`Bx_prev`) is part of the inference state and must be carried across decode steps alongside the SSM state.

---

## Running the Sanity Checks

```bash
python3 mamba3.py
```

Runs SISO and MIMO forward passes, single-step decode, shape assertions, and a parameter count for a 24-layer 2B-parameter model (instantiated on `device="meta"` — no RAM allocated).

---

## Dependencies

```
torch
einops
```

---

## References

- **Paper:** Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- **Official implementation:** [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/main) — production code with Triton/CUDA/TileLang kernels
