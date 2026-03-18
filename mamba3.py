"""
Mamba-3: Improved Sequence Modeling using State Space Principles
================================================================
Paper: https://arxiv.org/abs/2603.15569
Authors: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

This is a clean, readable, from-scratch implementation that captures the three core ideas
introduced in Mamba-3, without any Triton/CUDA kernels or TileLang optimizations.

CORE IDEAS IN MAMBA-3:
=======================

1. EXPONENTIAL-TRAPEZOIDAL DISCRETIZATION
   - Classic Mamba used "Zero-Order Hold (exponential-Euler)" to convert the continuous SSM
     into a recurrence.  This is a first-order approximation and loses detail at large dt.
   - Mamba-3 uses the "trapezoidal" rule instead: it averages the B*x contribution at time
     t-1 and time t before applying the state decay.  This is a higher-order approximation
     and improves accuracy, especially for large step sizes.
   - Concretely:
       h_t  = exp(A * dt_t) * h_{t-1}  +  dt_t * trap_t * (B_t * x_t + B_{t-1} * x_{t-1}) / 2
       (trap_t is a learned sigmoid gate that blends between Euler and trapezoidal)

2. COMPLEX-VALUED (ROTARY) STATE SPACE
   - Standard SSMs keep a real-valued hidden state.  Real states cannot easily represent
     oscillatory / rotational patterns (e.g., parity of a running count).
   - Mamba-3 uses Rotary Position Embeddings (RoPE) applied to the B and C (key/query)
     projections.  This gives the state an effective complex-valued structure and lets it
     track rotational dependencies.
   - A small "angle" projection is learned per head, accumulated over time as a running sum
     scaled by dt, and then used to rotate B and C before the SSM update.

3. MULTI-INPUT MULTI-OUTPUT (MIMO) FORMULATION
   - Mamba-2 is SISO: one input vector x drives one output y via one SSM state.
   - During autoregressive decoding the GPU is memory-bandwidth bound, not compute bound.
   - MIMO reuses the *same* SSM state h to process R "parallel" copies of x (rank-R projections),
     turning the outer-product state update into a full matrix-multiply.  This multiplies
     FLOPs by R while keeping memory traffic constant => better hardware utilization.
   - The B and C projections also get rank-R counterparts (K and Q in the paper's attention
     analogy).

NOTATION CONVENTIONS (matching the paper and original code):
    B  = batch size
    L  = sequence length
    H  = number of SSM heads  (= d_inner / headdim)
    P  = headdim              (per-head feature dimension)
    D  = d_state              (SSM state size per head)
    R  = mimo_rank            (1 for SISO, >1 for MIMO)
    G  = num_bc_heads         (ngroups; B/C are shared across G heads)
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Helper: RMS Norm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Standard Root Mean Square Layer Normalization.

    Simpler than LayerNorm because it drops the mean-centering step.
    Formula:  y = x / rms(x) * weight,   where rms(x) = sqrt(mean(x^2) + eps)

    Used to normalize the B and C projections before the SSM update, which
    stabilizes training when the projections are large or have varying scale.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d) — normalize over the last dimension
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Helper: Rotary Embedding (RoPE) utilities
# ---------------------------------------------------------------------------

def build_rope_freqs(num_angles: int, device: torch.device) -> torch.Tensor:
    """Build the standard RoPE inverse-frequency vector.

    Each pair of dimensions (2i, 2i+1) rotates at frequency 1/10000^(2i/d).
    This gives the angle of rotation for one unit of 'time' (here scaled by dt).

    Returns:
        freqs: (num_angles,)  — one frequency per (pair of) state dimensions
    """
    # Standard RoPE schedule: theta_i = 1 / 10000^(2i / num_angles)
    i = torch.arange(num_angles, device=device, dtype=torch.float32)
    freqs = 1.0 / (10000.0 ** (i / num_angles))
    return freqs


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions of x by the given angles.

    x:       (..., 2 * num_angles) — the tensor to rotate
    angles:  (..., num_angles)     — rotation angles in radians

    The rotation of a 2-d pair (a, b) by angle θ gives:
        (a * cos θ - b * sin θ,  a * sin θ + b * cos θ)

    This is exactly what RoPE does in attention; here we apply it to B and C
    so they implicitly track complex-valued state transitions.
    """
    cos = torch.cos(angles)   # (..., num_angles)
    sin = torch.sin(angles)   # (..., num_angles)

    # Split x into even and odd pairs along the last dimension
    x1 = x[..., 0::2]   # even indices
    x2 = x[..., 1::2]   # odd indices

    # Rotate each pair
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos

    # Interleave back: (x1_rot, x2_rot, x3_rot, ...) → (e0,o0,e1,o1,...)
    out = torch.stack([x_rotated_1, x_rotated_2], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# Helper: SSM recurrence (core Mamba-3 scan)
# ---------------------------------------------------------------------------

def mamba3_siso_scan(
    x: torch.Tensor,      # (B, L, H, P)   — input values (V in attention analogy)
    B_proj: torch.Tensor, # (B, L, H, D)   — input projection (K, after RoPE + norm)
    C_proj: torch.Tensor, # (B, L, H, D)   — output projection (Q, after RoPE + norm)
    ADT: torch.Tensor,    # (B, L, H)      — log decay: A * dt (negative)
    DT: torch.Tensor,     # (B, L, H)      — time step dt (positive, after softplus)
    trap: torch.Tensor,   # (B, L, H)      — trapezoidal gate (sigmoid, in [0, 1])
    D_skip: torch.Tensor, # (H,)           — skip/residual weight
) -> torch.Tensor:
    """Pure-Python sequential SSM scan for SISO Mamba-3.

    This implements the recurrence:
        h_t = exp(A*dt_t) * h_{t-1}  +  dt_t * [trap_t * Bk_prev + (1-trap_t) * Bk_t] * x_t
        y_t = C_t @ h_t  +  D * x_t

    where:
      - exp(A*dt_t) is the state decay (computed from ADT = A*dt)
      - trap_t blends between the current and previous B*x contributions
        (trapezoidal integration: trap=0 → pure Euler; trap=1 → pure trapezoidal average)
      - D is the skip connection (x passes directly to output)

    NOTE: This sequential loop is correct but slow for long sequences.
    The original code uses Triton/CUDA parallel chunk scans for efficiency.
    For research/educational purposes this is easy to follow.

    Returns:
        y: (B, L, H, P)
    """
    B_batch, L, H, P = x.shape
    D_state = B_proj.shape[-1]
    device = x.device
    dtype = x.dtype

    # h: SSM hidden state  — shape (B_batch, H, P, D)
    # P dimensions of x are projected into a rank-D state for each head
    h = torch.zeros(B_batch, H, P, D_state, device=device, dtype=torch.float32)

    ys = []

    # B*x at the previous timestep (for trapezoidal integration)
    Bx_prev = torch.zeros(B_batch, H, P, D_state, device=device, dtype=torch.float32)

    for t in range(L):
        # Current inputs at position t
        x_t   = x[:, t]          # (B, H, P)
        B_t   = B_proj[:, t]     # (B, H, D)
        C_t   = C_proj[:, t]     # (B, H, D)
        adt_t = ADT[:, t]        # (B, H)   — A*dt, negative
        dt_t  = DT[:, t]         # (B, H)   — dt, positive
        tr_t  = trap[:, t]       # (B, H)   — sigmoid gate

        # State decay factor: exp(A*dt) — shape (B, H, 1, 1) for broadcasting
        decay = torch.exp(adt_t).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        # B_t * x_t outer product: shape (B, H, P, D)
        # Each head: for each feature-dim p and state-dim d, add x[p] * B[d]
        Bx_curr = torch.einsum("bhp,bhd->bhpd", x_t.float(), B_t.float())

        # dt scaled, trapezoidal blend of current and previous B*x
        # trap=0: only use current Bx (Euler/ZOH)
        # trap=1: average current and previous Bx (trapezoidal)
        dt_expanded = dt_t.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
        tr_expanded = tr_t.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)

        Bx_blended = (1.0 - tr_expanded) * Bx_curr + tr_expanded * 0.5 * (Bx_curr + Bx_prev)

        # State update: h_t = decay * h_{t-1} + dt * Bx_blended
        h = decay * h + dt_expanded * Bx_blended

        # Output: y_t = sum_d (C_t[d] * h_t[:, :, :, d])  +  D * x_t
        # C_t: (B, H, D) — contracts over state dimension D with h (B, H, P, D)
        y_t = torch.einsum("bhd,bhpd->bhp", C_t.float(), h)  # (B, H, P)
        y_t = y_t + D_skip.unsqueeze(0).unsqueeze(-1) * x_t.float()  # skip conn

        ys.append(y_t.to(dtype))
        Bx_prev = Bx_curr

    # Stack along sequence dimension
    y = torch.stack(ys, dim=1)  # (B, L, H, P)
    return y


def mamba3_mimo_scan(
    x: torch.Tensor,       # (B, L, H, P)    — input values
    B_proj: torch.Tensor,  # (B, L, R, H, D) — K projections (R rank copies)
    C_proj: torch.Tensor,  # (B, L, R, H, D) — Q projections (R rank copies)
    ADT: torch.Tensor,     # (B, L, H)       — log decay
    DT: torch.Tensor,      # (B, L, H)       — time step
    trap: torch.Tensor,    # (B, L, H)       — trapezoidal gate
    D_skip: torch.Tensor,  # (H,)            — skip weight
    mimo_x: torch.Tensor,  # (H, R, P)       — MIMO down-project for x
    mimo_o: torch.Tensor,  # (H, R, P)       — MIMO up-project for output
) -> torch.Tensor:
    """MIMO variant of the Mamba-3 SSM scan.

    MIMO replaces the full outer-product state (P×D) used in SISO with a
    lower-dimensional shared state (D) updated by R rank-1 contributions.
    This trades the per-token outer product (P×D write) for a sum of R
    scalar-times-vector terms — the key hardware-efficiency win.

    Shapes:
        SISO state h: (B, H, P, D)  — headdim × d_state
        MIMO state h: (B, H, D)     — just d_state (P is projected away)

    State update per timestep t:
        x_r   = einsum("bhp,hrp->bhr", x_t, mimo_x)  # project x to R scalars per head
        Bx_t  = einsum("bhr,brhd->bhd", x_r, B_t)    # sum of R outer contributions → (B,H,D)
        h_t   = exp(A·dt) * h_{t-1} + dt * blend(Bx_t, Bx_prev)  # scalar * D-vec

    Output update per timestep t:
        y_r   = einsum("brhd,bhd->brh", C_t, h_t)    # R scalars per head (B, R, H)
        skip  = D * x_r                                # skip connection (B, H, R)
        y_t   = einsum("brh,hrp->bhp", y_r+skip, mimo_o)  # up-project to headdim

    Returns:
        y: (B, L, H, P)
    """
    B_batch, L, H, P = x.shape
    R = B_proj.shape[2]
    D_state = B_proj.shape[-1]
    device = x.device
    dtype = x.dtype

    # MIMO state is just D-dimensional (no P dimension — P is projected away)
    h      = torch.zeros(B_batch, H, D_state, device=device, dtype=torch.float32)
    Bx_prev = torch.zeros(B_batch, H, D_state, device=device, dtype=torch.float32)

    ys = []

    for t in range(L):
        x_t   = x[:, t]      # (B, H, P)
        B_t   = B_proj[:, t] # (B, R, H, D)
        C_t   = C_proj[:, t] # (B, R, H, D)
        adt_t = ADT[:, t]    # (B, H)
        dt_t  = DT[:, t]     # (B, H)
        tr_t  = trap[:, t]   # (B, H)

        decay = torch.exp(adt_t)   # (B, H)
        dt_e  = dt_t               # (B, H)
        tr_e  = tr_t               # (B, H)

        # ── Down-project x from headdim P to R rank-scalars per head ──────────
        # x_r[b, h, r] = dot(x_t[b, h, :], mimo_x[h, r, :])  — scalar per head per rank
        x_r = torch.einsum("bhp,hrp->bhr", x_t.float(), mimo_x.float())  # (B, H, R)

        # ── Accumulate state contribution across all R ranks ──────────────────
        # For rank r: contribution = x_r[b,h,r] * B_t[b,r,h,:]  — (B, H, D)
        # Bx_curr = sum_r contribution                              — (B, H, D)
        Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_t.float())  # (B, H, D)

        # ── Trapezoidal blend ─────────────────────────────────────────────────
        # trap=0 → pure current (Euler); trap=1 → average current+previous (trapezoid)
        tr_e3 = tr_e.unsqueeze(-1)            # (B, H, 1)
        Bx_blended = (1.0 - tr_e3) * Bx_curr + tr_e3 * 0.5 * (Bx_curr + Bx_prev)

        # ── State update: scalar multiply (no P dim needed) ───────────────────
        # h: (B, H, D);  decay and dt_e: (B, H) — unsqueeze for broadcast
        h = decay.unsqueeze(-1) * h + dt_e.unsqueeze(-1) * Bx_blended  # (B, H, D)

        # ── Per-rank output scalar (before up-projection) ─────────────────────
        # y_r[b, r, h] = dot(C_t[b, r, h, :], h[b, h, :])  — scalar per rank per head
        y_r_scalar = torch.einsum("brhd,bhd->brh", C_t.float(), h)  # (B, R, H)

        # ── D skip connection (per rank scalar) ──────────────────────────────
        # skip[b, r, h] = D[h] * x_r[b, h, r]
        skip = D_skip.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)  # (B, R, H)

        # ── Up-project combined output to headdim P ───────────────────────────
        # y_t[b, h, p] = sum_r (y_r_scalar[b,r,h] + skip[b,r,h]) * mimo_o[h, r, p]
        y_pre = y_r_scalar + skip                                      # (B, R, H)
        y_t   = torch.einsum("brh,hrp->bhp", y_pre, mimo_o.float())   # (B, H, P)

        ys.append(y_t.to(dtype))
        Bx_prev = Bx_curr  # update trapezoidal memory

    y = torch.stack(ys, dim=1)  # (B, L, H, P)
    return y


# ---------------------------------------------------------------------------
# Main Mamba3 Module
# ---------------------------------------------------------------------------

class Mamba3(nn.Module):
    """Mamba-3 sequence mixing layer.

    Drop-in replacement for a Transformer attention layer.
    Input/output shape: (batch, seq_len, d_model).

    Key parameters
    --------------
    d_model:      Token embedding dimension (hidden size of the model)
    d_state:      SSM hidden state dimension per head (D in the paper)
    expand:       Inner dimension multiplier; d_inner = expand * d_model
    headdim:      Dimension per SSM head; nheads = d_inner / headdim
    ngroups:      Number of groups for B/C projections (shared across groups)
    rope_fraction: Fraction of d_state dimensions that are "rotary"
                   (0.5 → half the state dimensions rotate, i.e. num_rope_angles = d_state/4)
    dt_min/max:   Range for the initial time-step dt values
    is_mimo:      If True, use MIMO formulation with rank=mimo_rank
    mimo_rank:    Number of parallel MIMO streams (R in the paper)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # ── Dimensions ──────────────────────────────────────────────────────
        self.d_model   = d_model
        self.d_state   = d_state
        self.expand    = expand
        self.headdim   = headdim
        self.A_floor   = A_floor
        self.is_mimo   = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.num_bc_heads = ngroups   # B and C are shared across this many heads

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // headdim   # H: total number of SSM heads

        # ── RoPE / angle dimensions ─────────────────────────────────────────
        # rope_fraction controls what fraction of d_state dimensions use rotation.
        # 0.5 → the first d_state/2 dims are real/imaginary pairs → d_state/4 angles
        # 1.0 → all d_state dims rotate → d_state/2 angles
        assert rope_fraction in [0.5, 1.0], "Only rope_fraction ∈ {0.5, 1.0} supported"
        # split_tensor_size: how many state dims participate in rotation
        self.split_tensor_size = int(d_state * rope_fraction)
        # ensure even (pairs of 2 for cos/sin)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        # Number of rotation angles = half the rotating state dims
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # ── Input projection ─────────────────────────────────────────────────
        # Single linear that produces all projections at once:
        #   z:      d_inner      (gate for output)
        #   x:      d_inner      (input values, V in attention analogy)
        #   B:      d_state * ngroups * mimo_rank   (input proj / K)
        #   C:      d_state * ngroups * mimo_rank   (output proj / Q)
        #   dd_dt:  nheads       (raw time step logit)
        #   dd_A:   nheads       (raw state decay logit)
        #   trap:   nheads       (trapezoidal gate logit)
        #   angles: num_rope_angles  (per-head rotation angle)
        d_in_proj = (
            2 * self.d_inner
            + 2 * d_state * ngroups * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        # ── dt bias (initialized so softplus gives values in [dt_min, dt_max]) ──
        # We sample dt from a log-uniform distribution and compute the bias
        # as inv_softplus(dt) = dt + log(1 - exp(-dt)), so that softplus(bias) ≈ dt
        _dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # inverse softplus: bias = x + log(1 - exp(-x))  (since softplus(bias) ≈ x for small x)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True  # usually excluded from weight decay

        # ── B and C biases (scalar offsets added inside the norm) ─────────────
        # Initialized to 1 so that before training the B/C projections are
        # close to 1 (no suppression of the state update).
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.B_bias._no_weight_decay = True
        self.C_bias._no_weight_decay = True

        # ── RMS norms for B and C ────────────────────────────────────────────
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # ── MIMO projection matrices ─────────────────────────────────────────
        # mimo_x: projects x (headdim-dim) down to R scalar values per head
        # mimo_o: projects R scalar values back up to headdim per head
        # mimo_z: same down-projection for the gate z (used in output norm)
        # Initialized to 1/R so that the sum over ranks is approximately 1x.
        if self.is_mimo:
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )
            self.mimo_z = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs)
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs) / self.mimo_rank
            )

        # ── D skip connection ─────────────────────────────────────────────────
        # Simple learned per-head scalar that adds a direct x→y shortcut,
        # similar to the "D" term in classic SSMs.
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        # ── Output projection ─────────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (batch, seq_len, d_model)  — input token embeddings

        Returns:
            out: (batch, seq_len, d_model)  — same shape as u
        """
        batch, L, d_model = u.shape

        # ── Step 1: Single fused projection ──────────────────────────────────
        # All parameters come from one big linear to minimize memory traffic.
        zxBCdtAtrap = self.in_proj(u)   # (B, L, d_in_proj)

        # Split into named components
        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,                                        # z: gate
                self.d_inner,                                        # x: values (V)
                self.d_state * self.num_bc_heads * self.mimo_rank,   # B: keys (K)
                self.d_state * self.num_bc_heads * self.mimo_rank,   # C: queries (Q)
                self.nheads,                                         # raw dt
                self.nheads,                                         # raw A
                self.nheads,                                         # trap gate
                self.num_rope_angles,                                # rotation angles
            ],
            dim=-1,
        )

        # ── Step 2: Reshape to head-based tensors ─────────────────────────────
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)    # (B, L, H, P)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)    # (B, L, H, P)

        # B and C: reshape to (B, L, R, G, D)
        #   R = mimo_rank, G = num_bc_heads (groups), D = d_state
        B_raw = rearrange(B_raw, "b l (r g n) -> b l r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)
        C_raw = rearrange(C_raw, "b l (r g n) -> b l r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)

        # ── Step 3: Compute state decay A*dt and time step dt ─────────────────
        # A is forced negative (softplus then negate) and clamped away from 0.
        # Large |A*dt| → fast forgetting; small |A*dt| → slow forgetting.
        A = -F.softplus(dd_A.float())                  # (B, L, H), negative
        A = A.clamp(max=-self.A_floor)                 # keep magnitude ≥ A_floor
        DT = F.softplus(dd_dt.float() + self.dt_bias)  # (B, L, H), positive
        ADT = A * DT                                   # (B, L, H), negative

        # ── Step 4: Trapezoidal gate ──────────────────────────────────────────
        # trap=0: standard "Euler/ZOH" update (Mamba-2 style)
        # trap=1: full trapezoidal blend (averages current and previous B*x)
        trap = torch.sigmoid(trap_raw.float())   # (B, L, H), in [0, 1]

        # ── Step 5: Apply RMS norm to B and C, expand groups → heads, add bias ──
        # Normalizing B and C prevents the projections from growing unboundedly.
        # B_raw shape: (B, L, R, G, D)  where G = ngroups
        B_normed = self.B_norm(B_raw.float())   # (B, L, R, G, D)
        C_normed = self.C_norm(C_raw.float())   # (B, L, R, G, D)

        # Expand from G groups to H heads — requires G == 1 (each group shared
        # across all heads) or G == nheads (one group per head).
        B_exp = B_normed.expand(-1, -1, -1, self.nheads, -1)  # (B, L, R, H, D)
        C_exp = C_normed.expand(-1, -1, -1, self.nheads, -1)  # (B, L, R, H, D)

        # B_bias / C_bias shape: (H, R, D) → rearrange to (R, H, D) for broadcast
        B_bias_t = rearrange(self.B_bias, "h r d -> r h d")  # (R, H, D)
        C_bias_t = rearrange(self.C_bias, "h r d -> r h d")  # (R, H, D)
        B_exp = B_exp + B_bias_t  # (B, L, R, H, D) + (R, H, D) broadcasts correctly
        C_exp = C_exp + C_bias_t  # (B, L, R, H, D)

        # ── Step 6: Apply RoPE rotation to B and C ───────────────────────────
        # Cumulative angle = sum_{s≤t}(dt_s * angle_s), independently per head.
        # angle_raw: (B, L, num_rope_angles) — learned rotation rate per step
        # DT:        (B, L, H)              — per-head time step
        # angle_increments: (B, L, H, num_rope_angles) — dt-scaled angle per head per step
        angle_increments = (
            angle_raw.float().unsqueeze(2)   # (B, L, 1, S)
            * DT.float().unsqueeze(-1)       # (B, L, H, 1)
        )   # → (B, L, H, S)
        cumulative_angles = torch.cumsum(angle_increments, dim=1)  # (B, L, H, S)

        # Expand to (B, L, R, H, S) for rotation applied to all ranks equally
        angles_for_rot = cumulative_angles.unsqueeze(2).expand(
            batch, L, self.mimo_rank, self.nheads, self.num_rope_angles
        )  # (B, L, R, H, num_rope_angles)

        # Rotate only the first `split_tensor_size` state dims of B and C.
        # Remaining dims are left unrotated (real-valued).
        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)  # (..., split_tensor_size)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)

        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)  # (B, L, R, H, D)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)  # (B, L, R, H, D)

        # ── Step 7: SSM scan ─────────────────────────────────────────────────
        if self.is_mimo:
            # MIMO: state is (B, H, D) — P dimension is projected away via mimo_x
            y = mamba3_mimo_scan(
                x=x,
                B_proj=B_proj,   # (B, L, R, H, D)
                C_proj=C_proj,
                ADT=ADT,
                DT=DT,
                trap=trap,
                D_skip=self.D,
                mimo_x=self.mimo_x,
                mimo_o=self.mimo_o,
            )
            # Gate output with z using simple SiLU (matches non-outproj_norm path)
            y = y * F.silu(z.float())
        else:
            # SISO: squeeze out the R=1 rank dimension for the scan
            y = mamba3_siso_scan(
                x=x,
                B_proj=B_proj[:, :, 0],  # (B, L, H, D)
                C_proj=C_proj[:, :, 0],
                ADT=ADT,
                DT=DT,
                trap=trap,
                D_skip=self.D,
            )
            # Gate output with z using simple SiLU
            y = y * F.silu(z.float())

        # ── Step 8: Output projection ─────────────────────────────────────────
        # Flatten heads back to d_inner, then project to d_model
        y = rearrange(y, "b l h p -> b l (h p)")       # (B, L, d_inner)
        out = self.out_proj(y.to(x.dtype))              # (B, L, d_model)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Single-step recurrent decode (autoregressive inference)
    # ─────────────────────────────────────────────────────────────────────────

    def step(
        self,
        u: torch.Tensor,             # (batch, d_model)  — single new token
        angle_state: torch.Tensor,   # (batch, H, num_rope_angles)  — accumulated RoPE angles
        ssm_state: torch.Tensor,     # (batch, H, P, D)  — SSM hidden state h
        Bx_prev_state: torch.Tensor, # (batch, H, P, D)  — previous B*x (for trapezoidal)
    ):
        """Run a single autoregressive decode step.

        Maintains the same recurrence as the scan but for one timestep,
        updating the provided state tensors in-place.

        Returns:
            out:              (batch, d_model)
            angle_state:      updated (batch, H, num_rope_angles)
            ssm_state:        updated (batch, H, P, D)
            Bx_prev_state:    updated (batch, H, P, D)
        """
        batch = u.shape[0]

        # In-projection (no sequence dim here)
        zxBCdtAtrap = self.in_proj(u)   # (B, d_in_proj)

        (z, x, B_raw, C_raw,
         dd_dt, dd_A, trap_raw, angle_raw) = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )

        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)    # (B, H, P)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)    # (B, H, P)

        B_raw = rearrange(B_raw, "b (r g n) -> b r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)
        C_raw = rearrange(C_raw, "b (r g n) -> b r g n",
                          r=self.mimo_rank, g=self.num_bc_heads)

        A   = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)  # (B, H)
        DT  = F.softplus(dd_dt.float() + self.dt_bias)            # (B, H)
        ADT = A * DT                                               # (B, H)
        trap = torch.sigmoid(trap_raw.float())                     # (B, H)

        # ── RMS norm + expand groups to heads + add bias ─────────────────────
        # B_raw shape here (single step, no L dim): (B, R, G, D)
        B_normed = self.B_norm(B_raw.float())   # (B, R, G, D)
        C_normed = self.C_norm(C_raw.float())   # (B, R, G, D)
        # Expand G groups → H heads
        B_exp = B_normed.expand(-1, -1, self.nheads, -1)  # (B, R, H, D)
        C_exp = C_normed.expand(-1, -1, self.nheads, -1)  # (B, R, H, D)
        # Add bias: B_bias (H, R, D) → (R, H, D) for broadcast with (B, R, H, D)
        B_bias_t = rearrange(self.B_bias, "h r d -> r h d")  # (R, H, D)
        C_bias_t = rearrange(self.C_bias, "h r d -> r h d")  # (R, H, D)
        B_exp = B_exp + B_bias_t  # (B, R, H, D)
        C_exp = C_exp + C_bias_t  # (B, R, H, D)

        # ── RoPE: update cumulative angle state (per head) ───────────────────
        # delta_angle = angle_raw * dt  — per-head increment
        delta_angle = angle_raw.float().unsqueeze(1) * DT.float().unsqueeze(-1)
        # angle_state: (B, H, S); delta_angle: (B, H, S) — update in place for decode
        angle_state = angle_state + delta_angle           # (B, H, S)

        # Rotate B and C using updated cumulative angle; expand R dim
        angles_for_rot = angle_state.unsqueeze(1).expand(-1, self.mimo_rank, -1, -1)  # (B, R, H, S)
        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_for_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_for_rot)
        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)  # (B, R, H, D)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)  # (B, R, H, D)

        # ── State update (single timestep) ───────────────────────────────────
        decay = torch.exp(ADT)   # (B, H)
        tr    = trap             # (B, H)

        if self.is_mimo:
            # MIMO state: (B, H, D)  — x projected to R scalars, not full P-dim outer product
            x_r = torch.einsum("bhp,hrp->bhr", x.float(), self.mimo_x.float())  # (B, H, R)

            # Sum of rank-1 contributions: Bx_curr[b,h,d] = sum_r x_r[b,h,r] * B_proj[b,r,h,d]
            Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_proj.float())  # (B, H, D)

            tr_e = tr.unsqueeze(-1)                             # (B, H, 1)
            Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + Bx_prev_state)
            ssm_state  = decay.unsqueeze(-1) * ssm_state + DT.unsqueeze(-1) * Bx_blended  # (B, H, D)

            # Per-rank output scalars then up-project
            y_r_scalar = torch.einsum("brhd,bhd->brh", C_proj.float(), ssm_state)  # (B, R, H)
            skip       = self.D.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)  # (B, R, H)
            y_pre      = y_r_scalar + skip                                          # (B, R, H)
            y          = torch.einsum("brh,hrp->bhp", y_pre, self.mimo_o.float())  # (B, H, P)
            y          = y * F.silu(z.float())

            Bx_prev_state = Bx_curr
        else:
            # SISO state: (B, H, P, D) — full outer product
            Bx_curr = torch.einsum("bhp,bhd->bhpd", x.float(), B_proj[:, 0].float())  # (B, H, P, D)
            tr_e    = tr.unsqueeze(-1).unsqueeze(-1)            # (B, H, 1, 1)
            Bx_blended = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + Bx_prev_state)
            ssm_state  = decay.unsqueeze(-1).unsqueeze(-1) * ssm_state + DT.unsqueeze(-1).unsqueeze(-1) * Bx_blended

            y = torch.einsum("bhd,bhpd->bhp", C_proj[:, 0].float(), ssm_state)  # (B, H, P)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x.float()
            y = y * F.silu(z.float())

            Bx_prev_state = Bx_curr

        y = rearrange(y, "b h p -> b (h p)")    # (B, d_inner)
        out = self.out_proj(y.to(u.dtype))       # (B, d_model)
        return out, angle_state, ssm_state, Bx_prev_state

    # ─────────────────────────────────────────────────────────────────────────
    # Inference state allocation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def allocate_inference_cache(self, batch_size: int, device=None, dtype=None):
        """Allocate zero-initialized states for autoregressive inference.

        State shapes differ between SISO and MIMO:
          SISO: ssm_state is (batch, H, P, D) — full headdim × d_state outer-product state
          MIMO: ssm_state is (batch, H, D)    — shared D-dimensional state (P projected away)

        Both modes share the same Bx_prev_state shape as ssm_state
        (trapezoidal integration memory).

        Returns:
            angle_state:   (batch, H, num_rope_angles)    — float32
            ssm_state:     (batch, H, P, D) or (batch, H, D)
            Bx_prev_state: same shape as ssm_state
        """
        device = device or self.in_proj.weight.device

        angle_state = torch.zeros(
            batch_size, self.nheads, self.num_rope_angles,
            device=device, dtype=torch.float32,
        )
        if self.is_mimo:
            # MIMO: state has no P (headdim) dimension — x is projected to R scalars
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.d_state,
                device=device, dtype=torch.float32,
            )
            Bx_prev_state = torch.zeros(
                batch_size, self.nheads, self.d_state,
                device=device, dtype=torch.float32,
            )
        else:
            # SISO: state is full outer product (headdim × d_state)
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )
            Bx_prev_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )
        return angle_state, ssm_state, Bx_prev_state

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_inner={self.d_inner}, nheads={self.nheads}, "
            f"headdim={self.headdim}, is_mimo={self.is_mimo}, "
            f"mimo_rank={self.mimo_rank}, num_rope_angles={self.num_rope_angles}"
        )


# ---------------------------------------------------------------------------
# Full stacked model
# ---------------------------------------------------------------------------

@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0          # >0 → add MLP sub-layer after each SSM block
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)    # passed to Mamba3(**ssm_cfg)
    attn_layer_idx: list = field(default_factory=list)  # reserved / not used here
    attn_cfg: dict = field(default_factory=dict)         # reserved / not used here
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True


class MambaBlock(nn.Module):
    """Single Mamba-3 residual block: Norm → Mamba3 → residual add."""

    def __init__(self, d_model: int, ssm_cfg: dict, device=None, dtype=None):
        super().__init__()
        self.norm  = RMSNorm(d_model)
        self.mixer = Mamba3(d_model=d_model, **ssm_cfg, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MLP(nn.Module):
    """SwiGLU-style feed-forward layer (used when d_intermediate > 0)."""

    def __init__(self, d_model: int, d_intermediate: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model,     bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * val)


class MambaLMHeadModel(nn.Module):
    """Full language-model built from stacked Mamba-3 blocks.

    Architecture:
        Embedding  →  n_layer × (MambaBlock [+ MLPBlock])  →  RMSNorm  →  LM head

    The LM head weight is optionally tied to the embedding weight.
    Vocab size is padded up to the nearest multiple of pad_vocab_size_multiple.
    """

    def __init__(self, config: MambaConfig, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

        # Pad vocab size
        vocab_size = config.vocab_size
        r = vocab_size % config.pad_vocab_size_multiple
        if r != 0:
            vocab_size += config.pad_vocab_size_multiple - r
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model, **factory_kwargs)

        self.layers = nn.ModuleList([
            MambaBlock(config.d_model, config.ssm_cfg, **factory_kwargs)
            for _ in range(config.n_layer)
        ])

        if config.d_intermediate > 0:
            self.mlp_norms  = nn.ModuleList([RMSNorm(config.d_model) for _ in range(config.n_layer)])
            self.mlp_layers = nn.ModuleList([
                MLP(config.d_model, config.d_intermediate, **factory_kwargs)
                for _ in range(config.n_layer)
            ])
        else:
            self.mlp_norms  = None
            self.mlp_layers = None

        self.norm_f  = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(vocab_size, config.d_model, bias=False, **factory_kwargs)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) → logits: (batch, seq_len, vocab_size)"""
        x = self.embedding(input_ids)
        for i, block in enumerate(self.layers):
            x = block(x)
            if self.mlp_layers is not None:
                x = x + self.mlp_layers[i](self.mlp_norms[i](x))
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Parameter counting utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module):
    """Return (trainable_params, total_params) for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
# Quick sanity-check (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Test SISO ─────────────────────────────────────────────────────────
    print("Testing SISO Mamba-3 …")
    model_siso = Mamba3(
        d_model=256,
        d_state=64,
        expand=2,
        headdim=32,
        ngroups=1,
        is_mimo=False,
    )
    x = torch.randn(2, 16, 256)
    y = model_siso(x)
    assert y.shape == x.shape, f"SISO shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ forward  output shape: {y.shape}")

    # Single-step decode
    angle_s, h_s, bx_s = model_siso.allocate_inference_cache(2)
    u_single = torch.randn(2, 256)
    o, angle_s, h_s, bx_s = model_siso.step(u_single, angle_s, h_s, bx_s)
    assert o.shape == (2, 256), f"SISO step shape mismatch: {o.shape}"
    print(f"  ✓ step     output shape: {o.shape}")

    trainable, total = count_parameters(model_siso)
    print(f"  Parameters — trainable: {trainable:,}  |  total: {total:,}")

    # ── Test MIMO ─────────────────────────────────────────────────────────
    print("Testing MIMO Mamba-3 …")
    model_mimo = Mamba3(
        d_model=256,
        d_state=64,
        expand=2,
        headdim=32,
        ngroups=1,
        is_mimo=True,
        mimo_rank=2,
    )
    y2 = model_mimo(x)
    assert y2.shape == x.shape, f"MIMO shape mismatch: {y2.shape} vs {x.shape}"
    print(f"  ✓ forward  output shape: {y2.shape}")

    # MIMO step: ssm_state is (B, H, D) — note no headdim dim (P projected away)
    angle_m, h_m, bx_m = model_mimo.allocate_inference_cache(2)
    assert h_m.shape == (2, model_mimo.nheads, model_mimo.d_state), \
        f"MIMO state shape mismatch: {h_m.shape}"
    o2, angle_m, h_m, bx_m = model_mimo.step(u_single, angle_m, h_m, bx_m)
    assert o2.shape == (2, 256), f"MIMO step shape mismatch: {o2.shape}"
    print(f"  ✓ step     output shape: {o2.shape}")
    print(f"  ✓ ssm_state shape (MIMO, no P-dim): {h_m.shape}")

    print("\nAll checks passed!")

    # ── Test full MambaLMHeadModel with the given MambaConfig ─────────────
    print("\nBuilding full MambaLMHeadModel (device='meta' — no memory allocated) …")
    cfg = MambaConfig(
        d_model=2048,
        d_intermediate=0,
        n_layer=24,
        vocab_size=50277,
        ssm_cfg={},          # use Mamba3 defaults
        attn_layer_idx=[],
        attn_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    )
    # Use device="meta" so no RAM is allocated for the ~2.7 B-param model.
    model_full = MambaLMHeadModel(cfg, device="meta")
    trainable_full, total_full = count_parameters(model_full)
    padded_vocab = model_full.vocab_size
    mamba3_defaults = Mamba3.__init__.__doc__ and "" or ""
    # Derive key Mamba3 dimensions for summary
    _m = Mamba3(d_model=cfg.d_model, **cfg.ssm_cfg, device="meta")
    print(f"  Config summary:")
    print(f"    d_model        = {cfg.d_model}")
    print(f"    n_layer        = {cfg.n_layer}")
    print(f"    vocab_size     = {cfg.vocab_size}  (padded → {padded_vocab})")
    print(f"    d_inner        = {_m.d_inner}")
    print(f"    nheads (H)     = {_m.nheads}")
    print(f"    headdim (P)    = {_m.headdim}")
    print(f"    d_state (D)    = {_m.d_state}")
    print(f"    is_mimo        = {_m.is_mimo}")
    print(f"    tie_embeddings = {cfg.tie_embeddings}")
    print(f"  Parameters — trainable: {trainable_full:,}  |  total: {total_full:,}")
    print(f"  Approx size (fp32): {total_full * 4 / 1e9:.2f} GB")
    print(f"  Approx size (bf16): {total_full * 2 / 1e9:.2f} GB")
