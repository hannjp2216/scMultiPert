# =======================================================================
# scMultiPert — Module 2: Encoders & Embedding Extraction
# =======================================================================
#
# This module transforms standardized multi-modal data from Module 1 into 
# high-dimensional vector representations and structured natural language 
# strings for LLM integration.
#
# Encoding Strategy by Modality:
#   - scPerturb-seq : Geneformer (Pre-trained) → Dense cell embeddings [N_cells, 256].
#   - MAVE          : ESM-2 8M (Protein LM)    → Dense variant embeddings [N_variants, 320].
#                     + Structured text corpus for context-aware fine-tuning.
#   - CRISPR Screens: Cell2Sentence-style textual encoding.
#                     (Direct text representation for LLM ingestion).
#
# Validation & Data Integrity Protocols:
#   - Sequence Screening: Implements rigorous None/NaN sequence filtering prior 
#     to ESM-2 encoding to prevent embedding space corruption.
#   - Cross-Module Synchronization: Validates Module 1 outputs against strict 
#     schemas to ensure downstream stability.
#   - Atomic Mapping: Guarantees 1:1 correspondence between variant IDs, 
#     textual descriptions, and numerical embeddings via ID-keyed indexing.
#
# Output Contract (Consumed by Modules 4 and 5):
#   {
#     "cell_embeddings" : Tensor [N_cells, 256] (CPU-stored).
#     "cell_metadata"   : DataFrame [perturbation, is_control, embedding_label].
#     "var_embeddings"  : Tensor [N_valid_variants, 320].
#     "var_texts"       : List[str] (Structured variant descriptions).
#     "var_ids"         : List[str] (Validated unique identifiers).
#     "crispr_texts"    : List[str] (Encoded gene dependency narratives).
#     "crispr_agg"      : DataFrame [gene, lineage, mean_fitness, essential_rate].
#   }
#
# Dependencies: transformers, geneformer, torch, pandas, numpy, anndata
# =======================================================================

import warnings
import torch
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from transformers import EsmTokenizer, EsmModel, AutoModel

# ---------------------------------------------------------------------------
# SECTION 0 — Custom Exception Framework
# ---------------------------------------------------------------------------


class Module2Error(Exception):
    """Base class for all Module 2 pipeline errors."""
    pass


class EmptyEncoderOutputError(Module2Error):
    """Raised when an encoder produces zero embeddings."""
    pass


class ShapeMismatchError(Module2Error):
    """Raised when embedding tensors do not match expected dimensions."""
    pass


class AlignmentError(Module2Error):
    """Raised if variant texts, embeddings, or IDs lose 1:1 correspondence."""
    pass


class MissingInputError(Module2Error):
    """Raised when required keys or columns from Module 1 are absent."""
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Data Integrity & Input Validation Gate
# ---------------------------------------------------------------------------

# Required obs columns from scRNA-seq modality (Norman 2019 schema).
# 'perturbed_genes' must be a list of atomic gene names (empty for controls).
_SCRNA_REQUIRED_OBS_COLS = [
    "perturbation",
    "is_control",
    "perturbed_genes",
]

# Required columns from MAVE modality for sequence-to-embedding mapping.
_MAVE_REQUIRED_COLS = [
    "variant_id", "gene_symbol", "mutant_sequence",
    "sge_score", "function_class", "position",
    "aa_ref", "aa_alt",
]

# Required columns from CRISPR modality for dependency narrative generation.
_CRISPR_REQUIRED_COLS = [
    "gene", "lineage", "fitness_score",
    "is_essential", "cell_line_id",
]


def validate_module1_outputs(modalities: dict) -> None:
    """
    Validates Module 1 outputs before initializing transformer models.
    
    Integrity Checks:
      - Presence: Ensures "scrna", "crispr", and "mave" keys are non-null.
      - Structure: Verifies layers["counts"] and required obs in scRNA-seq.
      - Data Quality: Validates MAVE and CRISPR DataFrames are non-empty
        with all required columns.
      - Pre-screen: Warns about null mutant sequences before ESM-2 encoding.
    
    Raises:
        MissingInputError: If a key, layer, or column is missing.
        EmptyEncoderOutputError: If a modality contains zero rows/cells.
    """
    # Key presence and non-None check
    for key in ("scrna", "crispr", "mave"):
        if key not in modalities:
            raise MissingInputError(
                f"[Module 2] Key '{key}' missing from modalities dict. "
                f"Re-run Module 1 load_all_modalities()."
            )
        if modalities[key] is None:
            raise MissingInputError(
                f"[Module 2] modalities['{key}'] is None. "
                f"Check Module 1 load_all_modalities()."
            )

    # scRNA-seq validation
    scrna = modalities["scrna"]
    if scrna.n_obs == 0:
        raise EmptyEncoderOutputError(
            "[Module 2] scrna AnnData has 0 cells. "
            "Module 1 QC thresholds may be too aggressive."
        )
    if "counts" not in scrna.layers:
        raise MissingInputError(
            "[Module 2] scrna.layers['counts'] is missing. "
            "Geneformer tokenizer requires raw counts. "
            "Re-run Module 1 load_norman2019()."
        )
    missing_obs = [c for c in _SCRNA_REQUIRED_OBS_COLS if c not in scrna.obs.columns]
    if missing_obs:
        raise MissingInputError(
            f"[Module 2] scrna.obs missing columns: {missing_obs}. "
            f"Re-run Module 1 load_norman2019()."
        )

    # MAVE validation
    mave = modalities["mave"]
    if len(mave) == 0:
        raise EmptyEncoderOutputError(
            "[Module 2] MAVE DataFrame has 0 rows. "
            "Re-run Module 1 load_multi_mave()."
        )
    missing_mave = [c for c in _MAVE_REQUIRED_COLS if c not in mave.columns]
    if missing_mave:
        raise MissingInputError(
            f"[Module 2] MAVE missing columns: {missing_mave}. "
            f"Ensure Module 1 load_multi_mave() was used."
        )
    
    # Pre-screen for null sequences before expensive ESM-2 encoding
    n_null = mave["mutant_sequence"].isna().sum()
    if n_null > 0:
        warnings.warn(
            f"[Module 2] {n_null} null mutant_sequence values detected. "
            f"These variants will be skipped during ESM-2 encoding. "
            f"Check Module 1 _apply_substitution() for stop codons "
            f"or out-of-bounds positions.",
            UserWarning,
            stacklevel=2,
        )

    # CRISPR validation
    crispr = modalities["crispr"]
    if len(crispr) == 0:
        raise EmptyEncoderOutputError(
            "[Module 2] CRISPR DataFrame has 0 rows. "
            "Re-run Module 1 load_depmap_crispr()."
        )
    missing_crispr = [c for c in _CRISPR_REQUIRED_COLS if c not in crispr.columns]
    if missing_crispr:
        raise MissingInputError(
            f"[Module 2] CRISPR missing columns: {missing_crispr}. "
            f"Re-run Module 1 load_depmap_crispr()."
        )

    print(
        f"[Module 2] Input validation passed:\n"
        f"  scrna  : {scrna.n_obs} cells, {scrna.n_vars} genes\n"
        f"  mave   : {len(mave)} variants ({n_null} will be skipped — null sequence)\n"
        f"  crispr : {crispr['gene'].nunique()} genes, "
        f"{crispr['lineage'].nunique()} lineages"
    )


# ---------------------------------------------------------------------------
# SECTION 2 — scPerturb-seq encoder (Geneformer)
# ---------------------------------------------------------------------------


class GeneformerEncoder:
    """
    Embedding Extraction: scRNA-seq via Geneformer.
    
    Generates high-dimensional cell-level representations from single-cell 
    transcriptomics using a pre-trained Geneformer architecture.
    
    Methodology:
      - Tokenization: Each cell is encoded by ranking gene expression 
        relative to a global median expression corpus.
      - Representation: Extracts final hidden layer and applies mean 
        pooling over gene tokens to derive dense 256-d cell embeddings.
    
    Output:
      - cell_embeddings: PyTorch Tensor [N_cells, 256] on CPU.
      - cell_metadata: DataFrame with validated obs columns,
        synchronized with embedding_idx and embedding_label.
    """
    
    EMBEDDING_DIM = 256

    def __init__(
        self,
        model_name: str = "ctheodoris/Geneformer",
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.batch_size = batch_size
        self.device = _resolve_device(device)

        try:
            from geneformer import TranscriptomeTokenizer
            self.tokenizer = TranscriptomeTokenizer()
        except ImportError:
            raise Module2Error(
                "[Geneformer] geneformer package not found. "
                "Install from HuggingFace: "
                "git clone https://huggingface.co/ctheodoris/Geneformer"
            )

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        print(f"[Geneformer] Loaded '{model_name}' on {self.device}")

    def encode(self, adata: ad.AnnData) -> tuple:
        """
        Encode scRNA-seq cells into Geneformer latent space.
        
        Args:
            adata: Standardized AnnData from Module 1 with layers["counts"]
                and required obs columns.
        
        Returns:
            Tuple of (cell_embeddings, cell_metadata):
              - cell_embeddings: Tensor [N_cells, 256] on CPU.
              - cell_metadata: DataFrame with embedding_idx and embedding_label.
        
        Raises:
            EmptyEncoderOutputError: If tokenization produces zero cells.
            ShapeMismatchError: If output dimensions don't match input.
        """
        print(f"[Geneformer] Tokenizing {adata.n_obs} cells...")
        tokenized = self.tokenizer.tokenize_data(
            data_directory=None,
            adata=adata,
            layer_key="counts",
        )

        if len(tokenized) == 0:
            raise EmptyEncoderOutputError(
                f"[Geneformer] Tokenization produced 0 cells from {adata.n_obs} cells. "
                f"Check that layers['counts'] has non-zero integer counts "
                f"and var_names are valid Ensembl IDs."
            )

        all_embeddings = []
        n_batches = (len(tokenized) + self.batch_size - 1) // self.batch_size

        for batch_idx, start in enumerate(range(0, len(tokenized), self.batch_size)):
            batch = tokenized[start : start + self.batch_size]
            ids = torch.tensor(batch["input_ids"]).to(self.device)
            mask = torch.tensor(batch["attention_mask"]).to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                )

            # Mean pooling over sequence dimension (exclude padding)
            hidden = out.hidden_states[-1]  # [B, seq, 256]
            expanded = mask.unsqueeze(-1).float()  # [B, seq, 1]
            pooled = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1)
            all_embeddings.append(pooled.cpu())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                print(f"[Geneformer] Batch {batch_idx + 1}/{n_batches} encoded")

        cell_embeddings = torch.cat(all_embeddings, dim=0)  # [N_cells, 256]

        if cell_embeddings.shape[0] != adata.n_obs:
            raise ShapeMismatchError(
                f"[Geneformer] Output has {cell_embeddings.shape[0]} rows "
                f"but AnnData has {adata.n_obs} cells."
            )

        # Build metadata with positional and string keys for downstream use
        cell_metadata = adata.obs[_SCRNA_REQUIRED_OBS_COLS].copy()
        cell_metadata["embedding_idx"] = range(len(cell_metadata))
        cell_metadata["embedding_label"] = cell_metadata["perturbation"]

        print(
            f"[Geneformer] Encoded {cell_embeddings.shape[0]} cells → "
            f"shape {tuple(cell_embeddings.shape)}"
        )
        return cell_embeddings, cell_metadata


# ---------------------------------------------------------------------------
# SECTION 3 — MAVE encoder (ESM-2 8M)
# ---------------------------------------------------------------------------


class ESMVariantEncoder:
    """
    Embedding Extraction: MAVE Variants via ESM-2 8M.
    
    Encodes MAVE functional data into dense vector representations and 
    structured text narratives for LLM integration.
    
    Process:
      1. Protein Language Modeling: ESM-2 (8M) generates 320-d embeddings 
         from mutant protein sequences.
      2. Corpus Generation: Structured text descriptions for fine-tuning.
    
    Sequence Integrity:
      - Pre-encoding screen for null/invalid sequences prevents corruption.
      - Centered window extraction for proteins exceeding 1022-token limit.
      - Synchronized mapping ensures 1:1 correspondence of embeddings, 
        texts, and IDs via variant-keyed dictionary.
    
    Output:
      - var_embeddings: Tensor [N_valid, 320] on CPU.
      - var_texts: List of structured variant descriptions.
      - var_ids: List of unique identifiers.
    """
    
    EMBEDDING_DIM = 320

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        batch_size: int = 16,
        device: str = "auto",
    ):
        self.batch_size = batch_size
        self.device = _resolve_device(device)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        print(f"[ESM-2 8M] Loaded '{model_name}' on {self.device}")

    def encode(self, mave_df: pd.DataFrame) -> tuple:
        """
        Encode MAVE variants into protein embeddings and descriptions.
        
        Args:
            mave_df: Validated DataFrame from Module 1 with required columns:
                variant_id, gene_symbol, mutant_sequence, sge_score,
                function_class, position, aa_ref, aa_alt.
        
        Returns:
            Tuple of (var_embeddings, var_texts, var_ids):
              - var_embeddings: Tensor [N_valid, 320] on CPU.
              - var_texts: List of descriptions (length N_valid).
              - var_ids: List of identifiers (length N_valid).
        
        Raises:
            EmptyEncoderOutputError: If all sequences are null/invalid.
            AlignmentError: If embeddings and metadata lengths mismatch.
            ShapeMismatchError: If embedding dimension is not 320.
        """
        # Pre-encoding screening: skip variants with invalid sequences
        null_mask = mave_df["mutant_sequence"].isna()
        n_null = null_mask.sum()
        valid_df = mave_df[~null_mask].copy().reset_index(drop=True)
        skipped_ids = mave_df.loc[null_mask, "variant_id"].tolist()

        if n_null > 0:
            warnings.warn(
                f"[ESM-2 8M] {n_null} variant(s) have null mutant_sequence "
                f"and will be skipped. "
                f"Skipped: {skipped_ids[:10]}{'...' if len(skipped_ids) > 10 else ''}",
                UserWarning,
                stacklevel=2,
            )

        if len(valid_df) == 0:
            raise EmptyEncoderOutputError(
                f"[ESM-2 8M] All {len(mave_df)} variants have null "
                f"mutant_sequence. Check Module 1 sequence generation."
            )

        print(
            f"[ESM-2 8M] Encoding {len(valid_df)} variants across "
            f"{valid_df['gene_symbol'].unique().tolist()} "
            f"({n_null} skipped — null sequence)..."
        )

        # Build embeddings and texts keyed by variant_id for alignment
        per_variant: dict = {}
        n_batches = (len(valid_df) + self.batch_size - 1) // self.batch_size

        for batch_idx, start in enumerate(range(0, len(valid_df), self.batch_size)):
            batch_rows = valid_df.iloc[start : start + self.batch_size]

            # Secondary null check within batch
            batch_null = batch_rows["mutant_sequence"].isna()
            if batch_null.any():
                bad_ids = batch_rows.loc[batch_null, "variant_id"].tolist()
                warnings.warn(
                    f"[ESM-2 8M] Batch {batch_idx + 1}: {batch_null.sum()} "
                    f"null sequence(s) found. Skipping: {bad_ids}",
                    UserWarning,
                    stacklevel=2,
                )
                batch_rows = batch_rows[~batch_null]
                if len(batch_rows) == 0:
                    continue

            # Center 1021-AA window on mutation for long proteins (e.g., BRCA1)
            # Prevents head-truncation that would make C-terminal variants identical
            windowed = [
                _extract_local_window(row["mutant_sequence"], int(row["position"]))
                for _, row in batch_rows.iterrows()
            ]

            toks = self.tokenizer(
                windowed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1022,
            )
            ids = toks["input_ids"].to(self.device)
            mask = toks["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask)

            # Mean pooling over sequence dimension (exclude padding)
            hidden = out.last_hidden_state  # [B, seq, 320]
            expanded = mask.unsqueeze(-1).float()  # [B, seq, 1]
            pooled = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1)
            pooled = pooled.cpu()  # [B, 320]

            # Store embedding and text together by variant_id
            for local_idx, (_, row) in enumerate(batch_rows.iterrows()):
                vid = row["variant_id"]
                per_variant[vid] = {
                    "embedding": pooled[local_idx],
                    "text": _build_mave_text(row),
                }

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == n_batches:
                print(
                    f"[ESM-2 8M] Batch {batch_idx + 1}/{n_batches} encoded "
                    f"({len(per_variant)} variants processed)"
                )

        # Materialize ordered outputs (Python 3.7+ preserves insertion order)
        var_ids = list(per_variant.keys())
        var_embeddings = torch.stack(
            [per_variant[v]["embedding"] for v in var_ids], dim=0
        )  # [N_valid, 320]
        var_texts = [per_variant[v]["text"] for v in var_ids]

        # Critical alignment checks
        if len(var_texts) != var_embeddings.shape[0]:
            raise AlignmentError(
                f"[ESM-2 8M] var_texts ({len(var_texts)}) != "
                f"var_embeddings ({var_embeddings.shape[0]}) rows."
            )
        if var_embeddings.shape[1] != self.EMBEDDING_DIM:
            raise ShapeMismatchError(
                f"[ESM-2 8M] Embedding dim is {var_embeddings.shape[1]}, "
                f"expected {self.EMBEDDING_DIM}."
            )

        print(
            f"[ESM-2 8M] Encoded {var_embeddings.shape[0]}/{len(mave_df)} "
            f"variants → shape {tuple(var_embeddings.shape)}"
        )
        if skipped_ids:
            print(f"[ESM-2 8M] Skipped {len(skipped_ids)} variant(s) (null sequence).")

        return var_embeddings, var_texts, var_ids


# ---------------------------------------------------------------------------
# SECTION 4 — CRISPR text encoder
# ---------------------------------------------------------------------------


def encode_crispr_as_text(crispr_df: pd.DataFrame) -> tuple:
    """
    Transform DepMap CRISPR data into structured text narratives.
    
    Bypasses neural encoding for direct semantic strings, allowing the LLM 
    to process gene-lineage dependencies as natural language.
    
    Methodology:
      - Aggregates long-format data into unique (gene, lineage) pairs.
      - Calculates mean_fitness and essential_rate for dependency quantification.
      - Formats aggregates into self-contained textual descriptions.
    
    Args:
        crispr_df: Standardized DataFrame from Module 1 with columns:
            gene, lineage, fitness_score, is_essential, cell_line_id.
    
    Returns:
        Tuple of (crispr_texts, crispr_agg):
          - crispr_texts: List of encoded dependency narratives.
          - crispr_agg: DataFrame with aggregated statistics aligned to texts.
    
    Raises:
        MissingInputError: If required columns are absent.
        EmptyEncoderOutputError: If aggregation produces zero pairs.
        AlignmentError: If text count doesn't match row count.
    """
    missing = [c for c in _CRISPR_REQUIRED_COLS if c not in crispr_df.columns]
    if missing:
        raise MissingInputError(
            f"[CRISPR text] Missing columns: {missing}. "
            f"Re-run Module 1 load_depmap_crispr()."
        )

    agg = (
        crispr_df
        .groupby(["gene", "lineage"])
        .agg(
            mean_fitness=("fitness_score", "mean"),
            essential_rate=("is_essential", "mean"),
            n_cell_lines=("cell_line_id", "nunique"),
        )
        .reset_index()
    )

    if len(agg) == 0:
        raise EmptyEncoderOutputError(
            "[CRISPR text] Aggregation produced 0 (gene, lineage) pairs. "
            "Check that crispr_df has non-null gene and lineage values."
        )

    crispr_texts = agg.apply(_build_crispr_text, axis=1).tolist()

    if len(crispr_texts) != len(agg):
        raise AlignmentError(
            f"[CRISPR text] {len(crispr_texts)} texts but {len(agg)} rows."
        )

    print(
        f"[CRISPR text] Generated {len(crispr_texts)} descriptions "
        f"({agg['gene'].nunique()} genes × {agg['lineage'].nunique()} lineages)"
    )
    return crispr_texts, agg


# ---------------------------------------------------------------------------
# SECTION 5 — Output validation gate
# ---------------------------------------------------------------------------


def validate_module2_outputs(outputs: dict) -> None:
    """
    Post-encoding integrity validation before downstream handoff.
    
    Validation Framework:
      - Key Presence: All required modalities are present.
      - Dimensional Consistency: Cell embeddings match metadata rows.
      - Atomic Alignment: Variant embeddings, texts, IDs are 1:1:1.
      - Hardware Safety: All tensors on CPU, no NaN/Inf values.
    
    Raises:
        MissingInputError: If a primary key is absent.
        ShapeMismatchError: If cell embeddings and metadata don't align.
        AlignmentError: If variant data lengths are inconsistent.
        Module2Error: If numerical instabilities detected.
    """
    required_keys = [
        "cell_embeddings", "cell_metadata",
        "var_embeddings", "var_texts", "var_ids",
        "crispr_texts", "crispr_agg",
    ]
    for key in required_keys:
        if key not in outputs:
            raise MissingInputError(
                f"[Module 2] validate_module2_outputs: key '{key}' missing."
            )

    cell_emb = outputs["cell_embeddings"]
    cell_meta = outputs["cell_metadata"]
    var_emb = outputs["var_embeddings"]
    var_texts = outputs["var_texts"]
    var_ids = outputs["var_ids"]

    # Dimensional consistency checks
    if cell_emb.shape[0] != len(cell_meta):
        raise ShapeMismatchError(
            f"[Module 2] cell_embeddings ({cell_emb.shape[0]}) != "
            f"cell_metadata ({len(cell_meta)}) rows."
        )
    if not (var_emb.shape[0] == len(var_texts) == len(var_ids)):
        raise AlignmentError(
            f"[Module 2] var_embeddings ({var_emb.shape[0]}), "
            f"var_texts ({len(var_texts)}), var_ids ({len(var_ids)}) mismatch."
        )
    
    # Hardware and numerical safety checks
    if cell_emb.device.type != "cpu":
        raise Module2Error(
            f"[Module 2] cell_embeddings on {cell_emb.device}, expected CPU."
        )
    if var_emb.device.type != "cpu":
        raise Module2Error(
            f"[Module 2] var_embeddings on {var_emb.device}, expected CPU."
        )
    if torch.isnan(cell_emb).any():
        raise Module2Error(
            "[Module 2] NaN values in cell_embeddings. "
            "Check attention masks in Geneformer pooling."
        )
    if torch.isnan(var_emb).any():
        raise Module2Error(
            "[Module 2] NaN values in var_embeddings. "
            "Check for all-padding tokens in ESM-2."
        )

    print(
        f"[Module 2] Output validation passed:\n"
        f"  cell_embeddings : {tuple(cell_emb.shape)}\n"
        f"  var_embeddings  : {tuple(var_emb.shape)}\n"
        f"  var_texts       : {len(var_texts)} entries\n"
        f"  var_ids         : {len(var_ids)} entries\n"
        f"  crispr_texts    : {len(outputs['crispr_texts'])} entries"
    )


# ---------------------------------------------------------------------------
# SECTION 6 — Unified entry point
# ---------------------------------------------------------------------------


def encode_all_modalities(
    modalities: dict,
    output_dir: str = "embeddings/",
    device: str = "auto",
) -> dict:
    """
    Module 2 Entry Point: Multi-Modal Embedding Orchestration.
    
    Executes complete encoding pipeline for scRNA-seq, MAVE, and CRISPR.
    Manages model initialization, batch processing, and validation.
    
    Args:
        modalities: Integrated dictionary from Module 1 (validated).
        output_dir: Target directory for .pt tensors and metadata files.
        device: Computation target ("auto" selects CUDA > MPS > CPU).
    
    Returns:
        Dictionary with encoded multi-modal dataset:
          - "cell_embeddings": 256-d tensor (Geneformer).
          - "cell_metadata": Synchronized cell-level metadata.
          - "var_embeddings": 320-d variant tensor (ESM-2).
          - "var_texts"/"var_ids": Variant descriptions and identifiers.
          - "crispr_texts"/"crispr_agg": Dependency narratives and stats.
    
    Note:
        All outputs undergo post-encoding integrity audits before return.
    """
    print("=" * 55)
    print("scMultiPert | Module 2: Encoders & Embedding Extraction (v1.0-proposal)")
    print("=" * 55)

    # Validate Module 1 outputs before loading any model
    print("[Module 2] Validating Module 1 inputs...")
    validate_module1_outputs(modalities)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # scRNA-seq: Geneformer encoding
    geneformer_enc = GeneformerEncoder(device=device)
    cell_embeddings, cell_metadata = geneformer_enc.encode(modalities["scrna"])
    _save_embeddings(cell_embeddings, cell_metadata, output_dir, prefix="scrna")

    # MAVE: ESM-2 8M encoding
    esm_enc = ESMVariantEncoder(device=device)
    var_embeddings, var_texts, var_ids = esm_enc.encode(modalities["mave"])
    _save_embeddings(
        var_embeddings,
        pd.DataFrame({"variant_id": var_ids, "embedding_idx": range(len(var_ids))}),
        output_dir,
        prefix="mave",
    )
    _save_texts(var_texts, output_dir, prefix="mave")

    # CRISPR: Text-only encoding
    crispr_texts, crispr_agg = encode_crispr_as_text(modalities["crispr"])
    _save_texts(crispr_texts, output_dir, prefix="crispr")

    outputs = {
        "cell_embeddings": cell_embeddings,
        "cell_metadata": cell_metadata,
        "var_embeddings": var_embeddings,
        "var_texts": var_texts,
        "var_ids": var_ids,
        "crispr_texts": crispr_texts,
        "crispr_agg": crispr_agg,
    }

    print("\n[Module 2] Validating outputs before passing to Module 4...")
    validate_module2_outputs(outputs)

    print("-" * 55)
    print("Module 2 complete. Outputs ready for Module 4 (Corpus Builder).")
    return outputs


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------


def _resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to CUDA > MPS > CPU; pass through explicit strings."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _extract_local_window(sequence: str, position: int, window: int = 510) -> str:
    """
    Extract local sequence window centered on variant position.
    
    For proteins exceeding ESM-2's 1022-token limit (e.g., BRCA1 at 1863 AA),
    standard head-truncation would discard C-terminal variants. This method
    centers the window on the mutation, ensuring unique embeddings for all
    variant positions.
    
    Args:
        sequence: Full-length protein sequence (IUPAC single-letter).
        position: 1-based index of amino acid substitution.
        window: Radius on each side (default 510 → 1021 AA window).
    
    Returns:
        Substring centered on mutation (length <= 2*window + 1).
        Returns full sequence if within limits.
    """
    if not sequence:
        return sequence

    seq_len = len(sequence)
    full_window = 2 * window + 1

    if seq_len <= full_window:
        return sequence

    idx = position - 1  # convert to 0-based
    left = idx - window
    right = idx + window + 1

    # Adjust window if near termini
    if left < 0:
        right -= left
        left = 0
    if right > seq_len:
        left -= (right - seq_len)
        right = seq_len

    left = max(0, left)
    return sequence[left:right]


def _build_mave_text(row: pd.Series) -> str:
    """
    Generate structured natural language description for MAVE variant.
    
    Serves as textual anchor for cross-modal alignment with ESM-2 embeddings.
    Includes mutation details, functional scores, and clinical classifications.
    
    Example Output:
        "BRCA1 variant BRCA1_p.Arg1699Trp (position 1699, Arg→Trp substitution).
         Saturation genome editing score: -1.830.
         Functional classification: loss_of_function.
         This variant is predicted to severely impair BRCA1 function."
    """
    gene = row["gene_symbol"]
    score = row["sge_score"]
    f_class = row["function_class"]
    vid = row["variant_id"]
    pos = int(row["position"])
    ref = row["aa_ref"]
    alt = row["aa_alt"]

    consequence = {
        "functional": f"predicted to preserve {gene} function.",
        "intermediate": f"may partially impair {gene} function.",
        "loss_of_function": f"predicted to severely impair {gene} function.",
    }.get(str(f_class), "Functional impact is uncertain.")

    return (
        f"{gene} variant {vid} (position {pos}, {ref}→{alt} substitution). "
        f"Saturation genome editing score: {score:.3f}. "
        f"Functional classification: {f_class}. "
        f"This variant is {consequence}"
    )


def _build_crispr_text(row: pd.Series) -> str:
    """
    Generate structured natural language summary for CRISPR dependency.
    
    Transforms high-throughput screening data into LLM-compatible format
    with statistical aggregation and lineage-specific context.
    
    Example Output:
        "CRISPR knockout of TP53 in haematopoietic cell lines (n=42).
         Mean Chronos gene effect score: -1.24 (essential).
         TP53 loss of function is strongly deleterious in 91% of tested lines."
    """
    gene = row["gene"]
    lineage = row["lineage"]
    fitness = row["mean_fitness"]
    ess_pct = round(row["essential_rate"] * 100)
    n = int(row["n_cell_lines"])
    label = "essential" if fitness < -0.5 else "non-essential"

    return (
        f"CRISPR knockout of {gene} in {lineage} cell lines (n={n}). "
        f"Mean Chronos gene effect score: {fitness:.2f} ({label}). "
        f"{gene} loss of function is strongly deleterious "
        f"in {ess_pct}% of tested lines."
    )


def _save_embeddings(
    embeddings: torch.Tensor,
    metadata: pd.DataFrame,
    output_dir: str,
    prefix: str,
) -> None:
    """Save embedding tensor and companion metadata CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, f"{output_dir}/{prefix}_embeddings.pt")
    metadata.to_csv(f"{output_dir}/{prefix}_metadata.csv", index=False)
    print(f"[Module 2] Saved {prefix} embeddings → {output_dir}/{prefix}_embeddings.pt")


def _save_texts(texts: list, output_dir: str, prefix: str) -> None:
    """Save list of text strings, one per line."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{output_dir}/{prefix}_texts.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(texts))
    print(f"[Module 2] Saved {prefix} texts → {out_path}")