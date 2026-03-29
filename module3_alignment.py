# =======================================================================
# scMultiPert — Module 3: Alignment Layer & Soft Token Projection
# =======================================================================
#
# This module implements the 'BiomedicalAligner' architecture, which projects 
# multi-modal dense embeddings into the LLM's latent space (4096-d). 
# It utilizes a Cross-Attention mechanism to fuse cellular state with 
# variant-level functional information.
#
# Neural Architecture:
#   - scRNA-seq Projection (Geneformer): 
#       LayerNorm(256) → MLP(256→2048→4096) → LayerNorm(4096).
#       Serves as the 'Query' in the cross-attention block.
#   - MAVE Projection (ESM-2): 
#       LayerNorm(320) → MLP(320→2048→4096) → LayerNorm(4096).
#       Serves as 'Key' and 'Value' in the cross-attention block.
#
# Multimodal Fusion Strategy:
#   - Mechanism: Single-head Cross-Attention (d_model=4096).
#   - Interaction: Enriches the cell embedding with variant-specific 
#     functional context, producing a dual-token soft sequence:
#       [Attended_Cell_Token, Projected_Variant_Token] -> Shape [B, 2, 4096].
#
# Design Principles:
#   - Synchronized Inference: 'forward_with_ids' ensures strict 1:1 pairing 
#     between (cell, variant) IDs during training and inference.
#   - Parameter Tracking: 'count_parameters' for experiment logging.
#   - Hardware Awareness: 'move_inputs_to_device' handles CPU→GPU transition.
#   - Dimensional Guards: Strict validation prevents silent broadcast errors.
#
# Public Interface (Consumed by Modules 5 and 6):
#   - BiomedicalAligner: Main architecture class.
#       .forward() / .forward_with_ids() : Core projection logic.
#       .freeze() / .unfreeze()          : Stage-1/2 control.
#       .count_parameters()              : Resource logging.
#   - Serializers: save_aligner() / load_aligner() for checkpointing.
#
# Dependencies: torch, torch.nn
# =======================================================================

import warnings
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# SECTION 0 — Custom Exception Framework
# ---------------------------------------------------------------------------


class Module3Error(Exception):
    """Base class for Module 3 domain-specific errors."""
    pass


class AlignerDimError(Module3Error):
    """Raised when input dimensions don't match architectural configuration."""
    pass


class IDSyncError(Module3Error):
    """Raised when embedding-metadata correspondence is compromised."""
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — ModalityProjector Architecture
# ---------------------------------------------------------------------------


class ModalityProjector(nn.Module):
    """
    Two-layer MLP mapping modality-specific latent spaces to LLM token space.
    
    Design Features:
      - Scale Stabilization: 'Double LayerNorm' normalizes variance between 
        low-dimensional encoders (256/320-d) and high-dimensional backbone (4096-d).
      - Non-linear Transformation: GELU activation for smoother gradient flow.
    
    Latent Flow:
      LayerNorm(In) → Linear(In→Hidden) → GELU → Dropout → Linear(Hidden→LLM)
      → LayerNorm(LLM)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        llm_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.llm_dim = llm_dim

        # Input stabilization layer
        self.norm_input = nn.LayerNorm(input_dim)
        
        # Core projection backbone
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
        )
        
        # LLM-space alignment layer
        self.norm_output = nn.LayerNorm(llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute stabilized projection.
        
        Args:
            x: Input tensor [Batch, input_dim].
        
        Returns:
            Projected tensor [Batch, llm_dim (4096)].
        """
        return self.norm_output(self.mlp(self.norm_input(x)))


# ---------------------------------------------------------------------------
# SECTION 2 — CrossModalAttention
# ---------------------------------------------------------------------------


class CrossModalAttention(nn.Module):
    """
    Single-Head Cross-Attention (SHCA) for cellular and variant fusion.
    
    Mathematical Basis:
      With Query (Cell) and Key/Value (Variant) at sequence length L=1,
      multi-head attention equals single linear projection. Using num_heads=1
      minimizes overhead while maintaining standard attention API.
    
    Information Flow:
      1. Query: Projected scRNA-seq embedding [B, 1, 4096]
      2. Key: Projected MAVE variant embedding [B, 1, 4096]
      3. Value: Projected MAVE variant embedding [B, 1, 4096]
    
    Architecture:
      MultiheadAttention(d=4096, h=1)
      → Residual: Add(Attended_Cell, Cell_Query)
      → LayerNorm(4096): Output stability for LLM injection.
    """
    
    def __init__(self, llm_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.llm_dim = llm_dim
        self.attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(llm_dim)

    def forward(
        self,
        cell_proj: torch.Tensor,  # [B, llm_dim]
        var_proj: torch.Tensor,   # [B, llm_dim]
    ) -> tuple:
        """
        Execute cross-attention fusion.
        
        Args:
            cell_proj: Projected cell embeddings [B, llm_dim].
            var_proj: Projected variant embeddings [B, llm_dim].
        
        Returns:
            Tuple of (attended_cell, attn_weights):
              - attended_cell: Cell enriched with variant context [B, llm_dim].
              - attn_weights: Attention weights for interpretability [B, 1, 1].
        """
        query = cell_proj.unsqueeze(1)  # [B, 1, 4096]
        key = var_proj.unsqueeze(1)   # [B, 1, 4096]
        value = var_proj.unsqueeze(1) # [B, 1, 4096]

        attn_out, attn_weights = self.attn(query, key, value)

        # Residual: preserve original cell signal while enriching with variant
        attended_cell = self.norm(attn_out.squeeze(1) + cell_proj)  # [B, 4096]
        return attended_cell, attn_weights


# ---------------------------------------------------------------------------
# SECTION 3 — BiomedicalAligner
# ---------------------------------------------------------------------------


class BiomedicalAligner(nn.Module):
    """
    Core integration module bridging biological encoders to LLM latent space.
    
    Orchestrates multi-modal projection, cross-modal attention, and soft-token 
    sequence generation for BioMistral-7B integration.
    
    Training Paradigm:
      - Stage 1 (Instruction Tuning): Aligner FROZEN via .freeze().
        LLM backbone adapts to biomedical domain through text instructions.
      - Stage 2 (Alignment Tuning): Aligner UNFROZEN via .unfreeze().
        Joint fine-tuning with BioMistral LoRA adapters.
    
    Architectural Specs (v1.0-proposal):
      - Cell Input: 256-d (matches GeneformerEncoder.EMBEDDING_DIM from Module 2).
      - Variant Input: 320-d (matches ESMVariantEncoder.EMBEDDING_DIM from Module 2).
      - Target Space: 4096-d (BioMistral-7B optimized).
    
    Input:
      - cell_emb [B, 256]: Dense cellular representation (scRNA-seq from Module 1/2).
      - var_emb [B, 320]: Variant functional embedding (MAVE from Module 1/2).
    
    Output [B, 2, 4096]:
      - Index 0: Attended Cell Token (cell state enriched with variant context).
      - Index 1: Raw Variant Token (direct projection of variant functional data).
      - Metadata: Attention weights [B, 1, 1] for interpretability.
    """
    
    def __init__(
        self,
        cell_emb_dim: int = 256,
        var_emb_dim: int = 320,
        hidden_dim: int = 2048,
        llm_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Store construction dims for checkpoint validation and logging
        self.cell_emb_dim = cell_emb_dim
        self.var_emb_dim = var_emb_dim
        self.hidden_dim = hidden_dim
        self.llm_dim = llm_dim

        self.proj_cell = ModalityProjector(cell_emb_dim, hidden_dim, llm_dim, dropout)
        self.proj_var = ModalityProjector(var_emb_dim, hidden_dim, llm_dim, dropout)
        self.cross_attn = CrossModalAttention(llm_dim, dropout)

    @property
    def device(self) -> torch.device:
        """Dynamic discovery of the module's active compute device."""
        return next(self.parameters()).device

    def move_inputs_to_device(
        self,
        cell_emb: torch.Tensor,
        var_emb: torch.Tensor,
    ) -> tuple:
        """
        Synchronize multi-modal inputs with aligner device.
        
        Module 2 encoders return CPU tensors for memory efficiency. This method
        ensures hardware-aware synchronization before GPU/MPS computation.
        Also consumed by Module 5's SoftTokenCollator for bulk pre-loading.
        
        Args:
            cell_emb: Cell embeddings (may be on CPU).
            var_emb: Variant embeddings (may be on CPU).
        
        Returns:
            Tuple of tensors moved to aligner device.
        
        Warns:
            If inputs arrive on mismatched devices (indicates upstream bug).
        """
        target = self.device

        if cell_emb.device != var_emb.device:
            warnings.warn(
                f"[BiomedicalAligner] cell_emb on {cell_emb.device}, "
                f"var_emb on {var_emb.device}. Moving both to {target}. "
                f"Check Module 2 returns all tensors on CPU.",
                UserWarning,
                stacklevel=3,
            )

        return cell_emb.to(target), var_emb.to(target)

    def _validate_input_dims(
        self,
        cell_emb: torch.Tensor,
        var_emb: torch.Tensor,
    ) -> None:
        """
        Verify input tensors are 2-D and match construction dimensions.
        
        Raises:
            AlignerDimError: If dimensions mismatch architectural config.
        """
        if cell_emb.ndim != 2:
            raise AlignerDimError(
                f"cell_emb must be 2-D [B, {self.cell_emb_dim}], "
                f"got {tuple(cell_emb.shape)}. "
                f"If single centroid, call .unsqueeze(0) first."
            )
        if var_emb.ndim != 2:
            raise AlignerDimError(
                f"var_emb must be 2-D [B, {self.var_emb_dim}], "
                f"got {tuple(var_emb.shape)}. "
                f"If single embedding, call .unsqueeze(0) first."
            )
        if cell_emb.shape[1] != self.cell_emb_dim:
            raise AlignerDimError(
                f"cell_emb dim is {cell_emb.shape[1]}, expected {self.cell_emb_dim}. "
                f"Module 2 GeneformerEncoder produces 256-d embeddings."
            )
        if var_emb.shape[1] != self.var_emb_dim:
            raise AlignerDimError(
                f"var_emb dim is {var_emb.shape[1]}, expected {self.var_emb_dim}. "
                f"Module 2 ESMVariantEncoder produces 320-d embeddings."
            )
        if cell_emb.shape[0] != var_emb.shape[0]:
            raise AlignerDimError(
                f"Batch size mismatch: cell_emb {cell_emb.shape[0]} vs "
                f"var_emb {var_emb.shape[0]}. "
                f"Module 5 SoftTokenCollator must use same batch indices."
            )

    def forward(
        self,
        cell_emb: torch.Tensor,  # [B, cell_emb_dim]
        var_emb: torch.Tensor,   # [B, var_emb_dim]
    ) -> tuple:
        """
        Standard forward pass for high-throughput training.
        
        Pipeline: Dim Validation → Device Sync → Projection → Fusion.
        
        Args:
            cell_emb: Dense cellular representations [B, 256].
            var_emb: Variant functional embeddings [B, 320].
        
        Returns:
            Tuple of (soft_tokens, attn_weights):
              - soft_tokens: Dual-token sequence [B, 2, 4096].
              - attn_weights: Cross-attention weights [B, 1, 1].
        """
        self._validate_input_dims(cell_emb, var_emb)
        cell_emb, var_emb = self.move_inputs_to_device(cell_emb, var_emb)

        # Modal-specific projection into LLM space (4096-d)
        cell_proj = self.proj_cell(cell_emb)  # [B, llm_dim]
        var_proj = self.proj_var(var_emb)     # [B, llm_dim]

        # Cross-modal synthesis
        attended_cell, attn_weights = self.cross_attn(cell_proj, var_proj)

        # Dual-Token Soft Sequence: [Attended_Cell, Raw_Variant]
        soft_tokens = torch.stack([attended_cell, var_proj], dim=1)  # [B, 2, 4096]
        
        return soft_tokens, attn_weights

    def forward_with_ids(
        self,
        cell_emb: torch.Tensor,  # [B, cell_emb_dim]
        var_emb: torch.Tensor,   # [B, var_emb_dim]
        cell_labels: list,       # list[str], len == B
        var_ids: list,           # list[str], len == B
    ) -> tuple:
        """
        Synchronized inference with metadata validation.
        
        Enforces 1:1 correspondence between embeddings and biological IDs.
        Used by Module 5 for training logs and Module 6 for inference validation.
        
        Args:
            cell_emb: Cell embeddings [B, 256].
            var_emb: Variant embeddings [B, 320].
            cell_labels: Cell identifiers (e.g., perturbation labels).
            var_ids: Variant identifiers (e.g., "BRCA1_p.Arg1699Trp").
        
        Returns:
            Tuple of (soft_tokens, attn_weights, pair_log):
              - soft_tokens: Dual-token sequence [B, 2, 4096].
              - attn_weights: Cross-attention weights [B, 1, 1].
              - pair_log: List of dicts with cell_label, var_id, attn_weight.
        
        Raises:
            IDSyncError: If metadata length doesn't match batch size.
        """
        B = cell_emb.shape[0]

        # Metadata batch synchronization check
        if len(cell_labels) != B:
            raise IDSyncError(
                f"cell_labels ({len(cell_labels)}) != batch size ({B})."
            )
        if len(var_ids) != B:
            raise IDSyncError(
                f"var_ids ({len(var_ids)}) != batch size ({B})."
            )

        # Core computation
        soft_tokens, attn_weights = self.forward(cell_emb, var_emb)

        # Build per-example synchronization log
        attn_scalars = attn_weights.detach().squeeze(-1).squeeze(-1).tolist()
        if isinstance(attn_scalars, float):
            attn_scalars = [attn_scalars]  # Handle B=1 edge case

        pair_log = [
            {
                "cell_label": cell_labels[i],
                "var_id": var_ids[i],
                "attn_weight": attn_scalars[i],
            }
            for i in range(B)
        ]

        return soft_tokens, attn_weights, pair_log

    def freeze(self) -> None:
        """
        Stage 1: Disable gradients for text-only instruction tuning.
        
        Projection layers remain static while LLM backbone adapts to 
        biomedical domain through text instructions.
        """
        for param in self.parameters():
            param.requires_grad = False
        
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[BiomedicalAligner] Frozen — Stage 1 (0 / {n_total:,} params trainable)")

    def unfreeze(self) -> None:
        """
        Stage 2: Enable gradients for multi-modal soft token alignment.
        
        Activated after Stage 1 convergence for joint fine-tuning with 
        BioMistral LoRA adapters (Module 5).
        """
        for param in self.parameters():
            param.requires_grad = True
        
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BiomedicalAligner] Unfrozen — Stage 2 ({n_trainable:,} params trainable)")

    def count_parameters(self) -> int:
        """
        Count trainable parameters for experiment tracking.
        
        Used by Module 5 (Weights & Biases) to monitor complexity and 
        ensure 'aligner_params' config is correctly logged.
        
        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# SECTION 4 — Serialization & Checkpoint Utilities
# ---------------------------------------------------------------------------


def save_aligner(
    aligner: BiomedicalAligner,
    output_dir: str,
    tag: str = "best",
) -> None:
    """
    Save aligner checkpoint with dimensional metadata.
    
    Packages state_dict with dims manifest for architecture reconstruction
    without hardcoded parameters. Enables version-safe loading.
    
    Args:
        aligner: Trained BiomedicalAligner instance.
        output_dir: Directory for checkpoint file.
        tag: Checkpoint identifier (default "best").
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / f"aligner_{tag}.pt"

    payload = {
        "state_dict": aligner.state_dict(),
        "dims": {
            "cell_emb_dim": aligner.cell_emb_dim,
            "var_emb_dim": aligner.var_emb_dim,
            "hidden_dim": aligner.hidden_dim,
            "llm_dim": aligner.llm_dim,
        },
    }
    torch.save(payload, save_path)
    print(
        f"[BiomedicalAligner] Saved checkpoint → {save_path}\n"
        f"  dims: cell_emb={aligner.cell_emb_dim}, "
        f"var_emb={aligner.var_emb_dim}, llm={aligner.llm_dim}"
    )


def load_aligner(
    checkpoint_path: str,
    cell_emb_dim: Optional[int] = None,
    var_emb_dim: Optional[int] = None,
    llm_dim: Optional[int] = None,
) -> BiomedicalAligner:
    """
    Load aligner checkpoint with dimension validation.
    
    Supports both v2 (metadata-aware) and v1 (legacy state_dict) formats.
    Proactive validation ensures weight compatibility with pipeline config.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        cell_emb_dim: Expected cell embedding dimension (optional).
        var_emb_dim: Expected variant embedding dimension (optional).
        llm_dim: Expected LLM dimension (optional).
    
    Returns:
        Reconstructed BiomedicalAligner in eval mode.
    
    Raises:
        AlignerDimError: If saved dimensions don't match expectations.
        Module3Error: If checkpoint format is unrecognized.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")

    # Format detection: v2 checkpoints contain "dims" manifest
    is_v2 = (
        isinstance(payload, dict)
        and "state_dict" in payload
        and "dims" in payload
    )

    if is_v2:
        saved_dims = payload["dims"]
        state_dict = payload["state_dict"]

        # Integrity check: caller expectations vs. saved architecture
        for dim_name, caller_val, saved_val in [
            ("cell_emb_dim", cell_emb_dim, saved_dims["cell_emb_dim"]),
            ("var_emb_dim", var_emb_dim, saved_dims["var_emb_dim"]),
            ("llm_dim", llm_dim, saved_dims["llm_dim"]),
        ]:
            if caller_val is not None and caller_val != saved_val:
                raise AlignerDimError(
                    f"Load mismatch: {dim_name} is {saved_val} in checkpoint, "
                    f"but {caller_val} was requested."
                )

        final_dims = {
            "cell_emb_dim": cell_emb_dim or saved_dims["cell_emb_dim"],
            "var_emb_dim": var_emb_dim or saved_dims["var_emb_dim"],
            "hidden_dim": saved_dims["hidden_dim"],
            "llm_dim": llm_dim or saved_dims["llm_dim"],
        }
        print(f"[BiomedicalAligner] Loaded v2 checkpoint ← {checkpoint_path}")

    elif isinstance(payload, dict):
        # Legacy v1 support: fallback to defaults or provided dims
        warnings.warn(
            f"[BiomedicalAligner] v1 checkpoint detected (no metadata). "
            f"Using fallback dimensions. Consider re-saving in v2 format.",
            UserWarning,
            stacklevel=2
        )
        state_dict = payload
        final_dims = {
            "cell_emb_dim": cell_emb_dim or 256,
            "var_emb_dim": var_emb_dim or 320,
            "hidden_dim": 2048,
            "llm_dim": llm_dim or 4096,
        }
    else:
        raise Module3Error(
            f"Unrecognized checkpoint format in '{checkpoint_path}'"
        )

    # Reconstruct and initialize
    aligner = BiomedicalAligner(**final_dims)
    aligner.load_state_dict(state_dict)
    aligner.eval()

    return aligner


# ---------------------------------------------------------------------------
# SECTION 5 — Architectural Profiling Utility
# ---------------------------------------------------------------------------


def print_parameter_summary() -> None:
    """
    Display aligner parameter distribution and overhead vs. BioMistral-7B.
    
    Useful for verifying alignment layer remains lightweight (~1% overhead)
    relative to the 7B parameter backbone.
    """
    aligner = BiomedicalAligner()

    sections = {
        "proj_cell": aligner.proj_cell,
        "proj_var": aligner.proj_var,
        "cross_attn": aligner.cross_attn,
    }

    total = 0
    print("\n[BiomedicalAligner] Parameter Summary (v1.0-proposal)")
    print("─" * 45)
    for name, module in sections.items():
        n = sum(p.numel() for p in module.parameters())
        total += n
        print(f"  {name:<14}: {n:>12,} params")
    print("─" * 45)
    print(f"  {'Total':<14}: {total:>12,} params")
    print(f"  {'BioMistral-7B':<14}: ~7,000,000,000 params")
    print(f"  Aligner overhead: ~{100 * total / 7e9:.1f}% of total system\n")


if __name__ == "__main__":
    # Baseline profiling on initialization
    print_parameter_summary()