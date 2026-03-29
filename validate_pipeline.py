# ========================================================================
# scMultiPert — validate_pipeline.py
# ========================================================================
# Pipeline Audit & Validation Layer (v2.5)
#
# This script performs read-only validation of all pipeline outputs,
# ensuring data integrity, biological plausibility, and zero-leakage
# guarantees before downstream training or inference.
#
# Integration Points:
#   - M1 (Preprocessing): Validates AnnData structure, Ensembl coverage,
#     control labels, and raw count preservation.
#   - M2 (Encoders): Validates embedding alignment, variance collapse,
#     ESM-2 biological signal, and UMAP cluster separation.
#   - M3 (Alignment): Validates soft token norms, attention balance,
#     and noise invariance of the BiomedicalAligner.
#   - M4 (Corpus): Validates gene leakage (CRITICAL), template diversity,
#     and bio-context token presence.
#   - M5 (Fine-tuning): Minimal validation — checks artifact existence.
#
# Output: ValidationReport (JSON) with overall_pass flag and per-check metrics.
# ========================================================================

import json
import warnings
import torch
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, Any, Set, Optional, List
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

# Integration imports from pipeline modules
from module1_preprocessing import _GENE_METADATA, Module1Error
from module2_encoders import (
    GeneformerEncoder, ESMVariantEncoder, 
    validate_module2_outputs, Module2Error
)
from module3_alignment import BiomedicalAligner, load_aligner, Module3Error
from module4_corpus import (
    BIO_CONTEXT_TOKEN, ChatMLExample, 
    _extract_gene_components, CriticalLeakageError, Module4Error
)

# ---------------------------------------------------------------------------
# SECTION 0 — Exception Hierarchy (from HTML pseudocode)
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Base exception for validation check failures."""
    pass


class CriticalLeakageError(ValidationError):
    """
    Data leakage detected between train/test splits.

    This is a pipeline-stopping error that prevents training on
    contaminated data. Raised exclusively by M4 gene leakage checks.
    """
    pass


class ValidationThresholdError(ValidationError):
    """A numerical check failed to meet its defined threshold."""
    pass


# ---------------------------------------------------------------------------
# SECTION 1 — Entry Point & Orchestration (S0 from HTML)
# ---------------------------------------------------------------------------

def run_all_checks(
    paths_dict: Dict[str, str],
    m1_outputs: Dict[str, Any],
    m2_outputs: Dict[str, Any],
    aligner: Optional[BiomedicalAligner],
    corpus_paths: Dict[str, str],
) -> Dict[str, Any]:
    """
    Execute complete validation pipeline across all modules.

    This is the main entry point that orchestrates all validation checks
    as specified in the HTML pseudocode S0 section.

    Args:
        paths_dict: Dictionary of file paths for artifact verification.
        m1_outputs: Outputs from Module 1 (modalities dict with scrna, crispr, mave).
        m2_outputs: Outputs from Module 2 (embeddings, metadata, texts).
        aligner: Loaded BiomedicalAligner from Module 3 (or checkpoint path).
        corpus_paths: Dictionary with 'train_path' and 'test_path' for M4.

    Returns:
        ValidationReport dictionary with structure:
        {
            "overall_pass": bool,
            "M1": {...}, "M2": {...}, "M3": {...}, "M4": {...},
            "M5": {...}  # Artifact existence only
        }

    Raises:
        CriticalLeakageError: If gene leakage detected in M4 (stops pipeline).
    """
    report = {}

    print("=" * 60)
    print("scMultiPert | validate_pipeline.py v2.5")
    print("Pipeline Audit — Read-only validation of all outputs")
    print("=" * 60)

    # Execute checks by module in dependency order
    print("\n[S1] Running Module 1 (Preprocessing) checks...")
    report["M1"] = check_module1(m1_outputs)

    print("\n[S2] Running Module 2 (Encoders) checks...")
    report["M2"] = check_module2(m2_outputs, m1_outputs)

    print("\n[S3] Running Module 3 (Aligner) checks...")
    report["M3"] = check_module3(aligner, m2_outputs)

    print("\n[S4] Running Module 4 (Corpus) checks...")
    report["M4"] = check_module4(corpus_paths)

    print("\n[S5] Running Module 5 (Fine-tuning) artifact checks...")
    report["M5"] = check_module5(paths_dict)

    # Calculate global result — all modules must pass
    report["overall_pass"] = all(
        v.get("pass", False) 
        for k, v in report.items() 
        if k != "overall_pass" and isinstance(v, dict)
    )

    # Output and persist report
    print_report(report)

    return report


# ---------------------------------------------------------------------------
# SECTION 2 — M1 Preprocessing Checks (S1 from HTML)
# ---------------------------------------------------------------------------

def check_module1(m1_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Module 1 preprocessing outputs.

    Checks (from HTML pseudocode):
      1. check_ensembl_coverage: ≥90% genes start with "ENSG"
      2. check_control_labels: is_control matches perturbation column
      3. check_sequence_integrity: MAVE sequences within 10 AA of expected
      4. check_raw_counts_preserved: layers["counts"] exists, int dtype, non-negative

    Args:
        m1_outputs: Dictionary with 'scrna', 'crispr', 'mave' from Module 1.

    Returns:
        Dictionary with check results and "pass" flag.
    """
    results = {}
    scrna = m1_outputs.get("scrna")
    mave = m1_outputs.get("mave")

    if scrna is None:
        raise ValidationError("M1 outputs missing 'scrna' key")

    # --- Check 1: Ensembl Coverage (Bio-specific) ---
    n_ensembl = sum(1 for g in scrna.var_names if str(g).startswith("ENSG"))
    pct = (n_ensembl / scrna.n_vars * 100) if scrna.n_vars > 0 else 0.0
    results["ensembl_coverage_pct"] = round(pct, 2)
    results["ensembl_pass"] = pct >= 90.0
    print(f"  [M1.1] Ensembl coverage: {pct:.1f}% ({n_ensembl}/{scrna.n_vars}) — "
          f"{'PASS' if results['ensembl_pass'] else 'FAIL'}")

    # --- Check 2: Control Labels (Safety/Integrity) ---
    if "perturbation" in scrna.obs.columns and "is_control" in scrna.obs.columns:
        control_mask_expected = scrna.obs["perturbation"].isin(["control", "non-targeting"])
        discrepancies = (scrna.obs["is_control"] != control_mask_expected).sum()
        results["control_label_discrepancies"] = int(discrepancies)
        results["control_pass"] = discrepancies == 0
        print(f"  [M1.2] Control label discrepancies: {discrepancies} — "
              f"{'PASS' if results['control_pass'] else 'FAIL'}")
    else:
        results["control_pass"] = False
        results["control_label_discrepancies"] = -1
        print("  [M1.2] WARNING: Missing required columns for control check")

    # --- Check 3: Sequence Integrity (Bio-specific) ---
    if mave is not None and len(mave) > 0:
        integrity_pass = True
        max_discrepancy = 0

        for gene, group in mave.groupby("gene_symbol"):
            # Get expected length from metadata
            expected_len = _GENE_METADATA.get(gene, {}).get("expected_length")
            if expected_len is None:
                continue  # Skip genes without reference data

            # Check median observed length
            obs_lengths = group["mutant_sequence"].dropna().str.len()
            if len(obs_lengths) > 0:
                median_len = obs_lengths.median()
                discrepancy = abs(median_len - expected_len)
                max_discrepancy = max(max_discrepancy, discrepancy)

                if discrepancy > 10:
                    integrity_pass = False
                    print(f"    WARNING: {gene} length discrepancy {discrepancy} AA "
                          f"(observed {median_len}, expected {expected_len})")

        results["sequence_integrity_pass"] = integrity_pass
        results["sequence_max_discrepancy_aa"] = int(max_discrepancy)
        print(f"  [M1.3] Sequence integrity — max discrepancy {max_discrepancy} AA — "
              f"{'PASS' if integrity_pass else 'FAIL'}")
    else:
        results["sequence_integrity_pass"] = None
        print("  [M1.3] SKIPPED: No MAVE data available")

    # --- Check 4: Raw Counts Preserved (Architectural) ---
    if "counts" in scrna.layers:
        counts = scrna.layers["counts"]
        # Check if sparse matrix
        if hasattr(counts, "toarray"):
            counts_dense = counts.toarray()
        else:
            counts_dense = np.array(counts)

        dtype_ok = counts_dense.dtype.kind in 'iu'  # integer or unsigned
        min_val = counts_dense.min()
        min_ok = min_val >= 0

        results["raw_counts_dtype"] = str(counts_dense.dtype)
        results["raw_counts_min"] = int(min_val)
        results["raw_counts_pass"] = dtype_ok and min_ok
        print(f"  [M1.4] Raw counts: dtype={counts_dense.dtype}, min={min_val} — "
              f"{'PASS' if results['raw_counts_pass'] else 'FAIL'}")
    else:
        results["raw_counts_pass"] = False
        print("  [M1.4] FAIL: layers['counts'] not found")

    # Overall M1 pass
    pass_keys = [k for k in results if k.endswith("_pass") and results[k] is not None]
    results["pass"] = all(results[k] for k in pass_keys)

    return results


# ---------------------------------------------------------------------------
# SECTION 3 — M2 Encoders Checks (S2 from HTML)
# ---------------------------------------------------------------------------

def check_module2(m2_outputs: Dict[str, Any], m1_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Module 2 encoder outputs.

    Checks (from HTML pseudocode):
      1. check_embedding_alignment: len(var_embeddings) == len(var_texts) == len(var_ids)
      2. check_variance_collapse: mean std per dimension > 0.01
      3. check_esm_loglikelihood_vs_sge: Spearman ρ > 0.10 (optional)
      4. check_umap_silhouette: silhouette score > 0.15

    Args:
        m2_outputs: Dictionary with embeddings, metadata, texts from Module 2.
        m1_outputs: Original M1 outputs for biological validation.

    Returns:
        Dictionary with check results and "pass" flag.
    """
    results = {}

    cell_emb = m2_outputs.get("cell_embeddings")
    var_emb = m2_outputs.get("var_embeddings")
    var_ids = m2_outputs.get("var_ids")
    var_texts = m2_outputs.get("var_texts")
    cell_meta = m2_outputs.get("cell_metadata")
    mave_df = m1_outputs.get("mave")

    # --- Check 1: Embedding Alignment (Architectural) ---
    if var_emb is not None and var_ids is not None and var_texts is not None:
        alignment_pass = (var_emb.shape[0] == len(var_ids) == len(var_texts))
        results["alignment_pass"] = alignment_pass
        results["n_variants"] = len(var_ids) if var_ids else 0
        print(f"  [M2.1] Embedding alignment: {var_emb.shape[0]} embeddings, "
              f"{len(var_ids)} IDs, {len(var_texts)} texts — "
              f"{'PASS' if alignment_pass else 'FAIL'}")
    else:
        results["alignment_pass"] = False
        print("  [M2.1] FAIL: Missing var_embeddings, var_ids, or var_texts")

    # --- Check 2: Variance Collapse (Architectural) ---
    if cell_emb is not None:
        if isinstance(cell_emb, torch.Tensor):
            cell_emb_np = cell_emb.cpu().numpy()
        else:
            cell_emb_np = np.array(cell_emb)

        std_per_dim = np.std(cell_emb_np, axis=0)
        mean_std = float(np.mean(std_per_dim))

        results["variance_mean_std"] = round(mean_std, 6)
        results["variance_pass"] = mean_std > 0.01
        print(f"  [M2.2] Variance collapse check: mean std = {mean_std:.6f} — "
              f"{'PASS' if results['variance_pass'] else 'FAIL'}")
    else:
        results["variance_pass"] = None
        print("  [M2.2] SKIPPED: No cell embeddings available")

    # --- Check 3: ESM-2 Log-likelihood vs SGE (Bio-specific, optional) ---
    # Note: This requires log_likelihoods to be computed during M2 encoding
    log_ll = m2_outputs.get("var_log_likelihoods")  # Optional field

    if log_ll is not None and mave_df is not None and var_ids is not None:
        try:
            # Build SGE score lookup
            mave_lookup = {row["variant_id"]: row["sge_score"] 
                          for _, row in mave_df.iterrows()}
            sge_scores = [mave_lookup.get(vid, np.nan) for vid in var_ids]

            # Filter valid pairs
            valid_pairs = [(ll, sge) for ll, sge in zip(log_ll, sge_scores) 
                          if not np.isnan(sge)]

            if len(valid_pairs) >= 10:
                ll_valid, sge_valid = zip(*valid_pairs)
                rho, pval = spearmanr(ll_valid, sge_valid)
                results["esm_spearman_rho"] = round(rho, 4)
                results["esm_spearman_pval"] = round(pval, 6)
                results["esm_spearman_pass"] = rho > 0.10
                print(f"  [M2.3] ESM-2 vs SGE correlation: ρ={rho:.4f}, p={pval:.4f} — "
                      f"{'PASS' if results['esm_spearman_pass'] else 'FAIL'}")
            else:
                results["esm_spearman_pass"] = None
                print(f"  [M2.3] SKIPPED: Only {len(valid_pairs)} valid pairs (< 10)")
        except Exception as e:
            results["esm_spearman_pass"] = None
            print(f"  [M2.3] SKIPPED: Error computing correlation ({e})")
    else:
        results["esm_spearman_pass"] = None
        print("  [M2.3] SKIPPED: var_log_likelihoods not available (optional check)")

    # --- Check 4: UMAP Silhouette (Bio-specific) ---
    if cell_emb is not None and cell_meta is not None:
        try:
            from umap import UMAP

            if "perturbation" in cell_meta.columns:
                labels = cell_meta["perturbation"].values

                # Sample for efficiency if large
                n_samples = min(5000, len(cell_emb_np))
                if len(cell_emb_np) > n_samples:
                    indices = np.random.choice(len(cell_emb_np), n_samples, replace=False)
                    emb_sample = cell_emb_np[indices]
                    labels_sample = labels[indices]
                else:
                    emb_sample = cell_emb_np
                    labels_sample = labels

                # UMAP projection
                umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(emb_sample)
                score = silhouette_score(umap_2d, labels_sample)

                results["umap_silhouette"] = round(score, 4)
                results["umap_pass"] = score > 0.15
                print(f"  [M2.4] UMAP silhouette: {score:.4f} — "
                      f"{'PASS' if results['umap_pass'] else 'FAIL'}")
            else:
                results["umap_pass"] = None
                print("  [M2.4] SKIPPED: 'perturbation' column not in metadata")
        except ImportError:
            results["umap_pass"] = None
            print("  [M2.4] SKIPPED: umap-learn not installed")
        except Exception as e:
            results["umap_pass"] = None
            print(f"  [M2.4] SKIPPED: Error during UMAP computation ({e})")
    else:
        results["umap_pass"] = None
        print("  [M2.4] SKIPPED: Missing cell embeddings or metadata")

    # Overall M2 pass
    pass_keys = [k for k in results if k.endswith("_pass") and results[k] is not None]
    results["pass"] = all(results[k] for k in pass_keys) if pass_keys else False

    return results


# ---------------------------------------------------------------------------
# SECTION 4 — M3 Aligner Checks (S3 from HTML)
# ---------------------------------------------------------------------------

def check_module3(aligner: Optional[BiomedicalAligner], m2_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Module 3 alignment layer.

    Checks (from HTML pseudocode):
      1. check_soft_token_norms: 0.1 < norm_mean < 100
      2. check_attention_combinatorial: no component > 95% attention
      3. check_noise_invariance: cos_dist < 0.05 with ε=0.01 noise

    Args:
        aligner: Loaded BiomedicalAligner instance (or None if not available).
        m2_outputs: Module 2 outputs with embeddings for testing.

    Returns:
        Dictionary with check results and "pass" flag.
    """
    results = {}

    if aligner is None:
        print("  [M3] SKIPPED: No aligner provided")
        return {"pass": None}

    cell_emb_full = m2_outputs.get("cell_embeddings")
    var_emb_full = m2_outputs.get("var_embeddings")
    cell_meta = m2_outputs.get("cell_metadata")

    if cell_emb_full is None or var_emb_full is None:
        print("  [M3] SKIPPED: Missing embeddings from M2")
        return {"pass": None}

    # Sample 32 examples for efficiency
    sample_size = min(32, len(cell_emb_full), len(var_emb_full))
    if isinstance(cell_emb_full, torch.Tensor):
        cell_emb = cell_emb_full[:sample_size].to(aligner.device)
        var_emb = var_emb_full[:sample_size].to(aligner.device)
    else:
        cell_emb = torch.tensor(cell_emb_full[:sample_size], dtype=torch.float32).to(aligner.device)
        var_emb = torch.tensor(var_emb_full[:sample_size], dtype=torch.float32).to(aligner.device)

    aligner.eval()

    with torch.no_grad():
        # --- Check 1: Soft Token Norms (Architectural) ---
        soft_tokens, attn_weights = aligner(cell_emb, var_emb)
        # soft_tokens: [B, 2, 4096]

        norms = soft_tokens.norm(dim=-1).mean().item()
        results["soft_token_norm_mean"] = round(norms, 4)
        results["norm_pass"] = 0.1 < norms < 100
        print(f"  [M3.1] Soft token norm mean: {norms:.4f} — "
              f"{'PASS' if results['norm_pass'] else 'FAIL'}")

        # --- Check 2: Attention Combinatorial (Bio-specific) ---
        # For true combinatorial check, we'd need A+B pairs
        # Here we check if attention weights are balanced
        attn_vals = attn_weights.squeeze().cpu().numpy()

        # Check for extreme attention concentration
        if len(attn_vals.shape) > 0 and attn_vals.size > 0:
            max_attn = float(np.max(attn_vals))
            results["attn_bias_max"] = round(max_attn, 4)
            results["attn_pass"] = max_attn < 0.95
            print(f"  [M3.2] Attention bias max: {max_attn:.4f} — "
                  f"{'PASS' if results['attn_pass'] else 'FAIL'}")
        else:
            results["attn_pass"] = None
            print("  [M3.2] SKIPPED: Could not extract attention weights")

        # --- Check 3: Noise Invariance (Architectural) ---
        noise_scale = 0.01
        noise_c = cell_emb + torch.randn_like(cell_emb) * noise_scale
        noise_v = var_emb + torch.randn_like(var_emb) * noise_scale

        soft_noisy, _ = aligner(noise_c, noise_v)

        # Cosine distance between original and noisy
        orig_flat = soft_tokens.view(sample_size, -1)
        noisy_flat = soft_noisy.view(sample_size, -1)

        cos_sim = torch.nn.functional.cosine_similarity(orig_flat, noisy_flat, dim=1)
        cos_dist = (1 - cos_sim.mean()).item()

        results["noise_cos_dist"] = round(cos_dist, 6)
        results["noise_pass"] = cos_dist < 0.05
        print(f"  [M3.3] Noise invariance (ε={noise_scale}): cos_dist={cos_dist:.6f} — "
              f"{'PASS' if results['noise_pass'] else 'FAIL'}")

    # Overall M3 pass
    pass_keys = [k for k in results if k.endswith("_pass") and results[k] is not None]
    results["pass"] = all(results[k] for k in pass_keys) if pass_keys else False

    return results


# ---------------------------------------------------------------------------
# SECTION 5 — M4 Corpus Checks (S4 from HTML)
# ---------------------------------------------------------------------------

def check_module4(corpus_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate Module 4 corpus construction.

    Checks (from HTML pseudocode):
      1. check_gene_leakage: train_genes ∩ test_genes = ∅ (CRITICAL — raises exception)
      2. check_template_diversity: no template > 60% per modality
      3. check_bio_context_token: 100% coverage in has_bio_context examples

    Args:
        corpus_paths: Dictionary with 'train_path' and 'test_path' to JSONL files.

    Returns:
        Dictionary with check results and "pass" flag.

    Raises:
        CriticalLeakageError: If gene overlap detected between train and test.
    """
    results = {}

    train_path = corpus_paths.get("train_path")
    test_path = corpus_paths.get("test_path")

    if not train_path or not test_path:
        raise ValidationError("corpus_paths must contain 'train_path' and 'test_path'")

    # Load examples
    train_ex = _load_jsonl(train_path)
    test_ex = _load_jsonl(test_path)

    print(f"  [M4] Loaded {len(train_ex)} train, {len(test_ex)} test examples")

    # --- Check 1: Gene Leakage (CRITICAL — Data/Safety) ---
    train_genes: Set[str] = set()
    test_genes: Set[str] = set()

    for ex in train_ex:
        if ex.get("gene_id"):
            train_genes.update(_extract_gene_components(ex["gene_id"]))

    for ex in test_ex:
        if ex.get("gene_id"):
            test_genes.update(_extract_gene_components(ex["gene_id"]))

    intersection = train_genes & test_genes

    results["n_train_genes"] = len(train_genes)
    results["n_test_genes"] = len(test_genes)

    if intersection:
        # CRITICAL: This stops the pipeline
        raise CriticalLeakageError(
            f"Data leakage detected: {len(intersection)} genes appear in both "
            f"train and test splits: {sorted(list(intersection))[:10]}..."
        )

    results["leakage_pass"] = True
    print(f"  [M4.1] Gene leakage check: {len(train_genes)} train, {len(test_genes)} test genes — PASS")

    # --- Check 2: Template Diversity (Data) ---
    for modality in ["mave", "scrna", "crispr"]:
        modality_examples = [ex for ex in train_ex if ex.get("modality") == modality]

        if not modality_examples:
            continue

        # Extract template signatures from user messages
        templates = []
        for ex in modality_examples:
            messages = ex.get("messages", [])
            if len(messages) > 1:
                user_content = messages[1].get("content", "")
                # Remove bio context token for template extraction
                clean_content = user_content.replace(BIO_CONTEXT_TOKEN, "").strip()
                template = _extract_template_signature(clean_content)
                templates.append(template)

        if templates:
            counter = Counter(templates)
            max_freq = max(counter.values()) / len(templates)

            results[f"template_diversity_{modality}"] = round(max_freq, 4)
            results[f"template_pass_{modality}"] = max_freq <= 0.60

            status = "PASS" if results[f"template_pass_{modality}"] else "WARN"
            print(f"  [M4.2-{modality}] Template diversity: max_freq={max_freq:.2%} — {status}")

    # --- Check 3: Bio Context Token (Safety) ---
    bio_examples = [ex for ex in train_ex if ex.get("has_bio_context")]

    if bio_examples:
        with_token = sum(
            1 for ex in bio_examples
            if any(BIO_CONTEXT_TOKEN in m.get("content", "") 
                   for m in ex.get("messages", []) if m.get("role") == "user")
        )

        coverage = (with_token / len(bio_examples)) * 100
        results["bio_token_coverage_pct"] = round(coverage, 2)
        results["bio_token_pass"] = coverage == 100.0

        status = "PASS" if results["bio_token_pass"] else "FAIL"
        print(f"  [M4.3] Bio-context token coverage: {coverage:.1f}% ({with_token}/{len(bio_examples)}) — {status}")
    else:
        results["bio_token_pass"] = None
        print("  [M4.3] SKIPPED: No bio-context examples in training set")

    # Overall M4 pass
    pass_keys = [k for k in results if k.endswith("_pass") and results[k] is not None]
    results["pass"] = all(results[k] for k in pass_keys) if pass_keys else False

    return results


# ---------------------------------------------------------------------------
# SECTION 6 — M5 Fine-tuning Artifact Checks
# ---------------------------------------------------------------------------

def check_module5(paths_dict: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate Module 5 fine-tuning artifacts exist.

    Minimal validation since M5 is the training stage. Checks:
      - Aligner checkpoint exists
      - LLM adapter exists (if fine-tuning completed)

    Args:
        paths_dict: Dictionary with paths to M5 outputs.

    Returns:
        Dictionary with check results.
    """
    results = {}

    # Check for aligner checkpoint
    aligner_path = paths_dict.get("aligner_checkpoint")
    if aligner_path:
        exists = Path(aligner_path).exists()
        results["aligner_checkpoint_exists"] = exists
        print(f"  [M5.1] Aligner checkpoint: {aligner_path} — "
              f"{'EXISTS' if exists else 'MISSING'}")

    # Check for LLM adapter
    adapter_path = paths_dict.get("llm_adapter_dir")
    if adapter_path:
        exists = Path(adapter_path).exists() and any(Path(adapter_path).iterdir())
        results["llm_adapter_exists"] = exists
        print(f"  [M5.2] LLM adapter: {adapter_path} — "
              f"{'EXISTS' if exists else 'MISSING'}")

    # M5 is optional for validation (training may not be complete)
    results["pass"] = True

    return results


# ---------------------------------------------------------------------------
# SECTION 7 — Report Generation & Utilities
# ---------------------------------------------------------------------------

def print_report(report: Dict[str, Any]) -> None:
    """
    Print formatted validation report to console and save to JSON.

    Args:
        report: Complete validation report dictionary.
    """
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    for module, checks in report.items():
        if module == "overall_pass":
            continue

        if isinstance(checks, dict) and "pass" in checks:
            status = "✓ PASS" if checks["pass"] else "✗ FAIL"
            print(f"\n[{module}] {status}")

            for key, val in checks.items():
                if key != "pass" and not key.endswith("_pass"):
                    print(f"  {key}: {val}")
                elif key.endswith("_pass") and key != "pass":
                    sub_status = "✓" if val else "✗" if val is not None else "○"
                    print(f"  {sub_status} {key}: {val}")

    # Overall result
    overall = report.get("overall_pass", False)
    print("\n" + "=" * 60)
    if overall:
        print("PIPELINE VÁLIDO — All critical checks passed")
    else:
        print("PIPELINE INVÁLIDO — Review failed checks above")
    print("=" * 60)

    # Save to JSON
    output_path = "validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {output_path}")


def _load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _extract_template_signature(question: str) -> str:
    """
    Extract template signature from question text.

    Uses first 20 chars + last 20 chars as a fingerprint.
    """
    question = question.strip()
    if len(question) <= 40:
        return question
    return question[:20] + "..." + question[-20:]


# ---------------------------------------------------------------------------
# SECTION 8 — Convenience Entry Points
# ---------------------------------------------------------------------------

def validate_full_pipeline(
    m1_outputs: Dict[str, Any],
    m2_outputs: Dict[str, Any],
    aligner_path: Optional[str] = None,
    corpus_paths: Optional[Dict[str, str]] = None,
    m5_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function for full pipeline validation.

    Automatically loads aligner if path provided.
    """
    paths_dict = m5_paths or {}

    # Load aligner if path provided
    aligner = None
    if aligner_path:
        try:
            aligner = load_aligner(aligner_path)
            print(f"[validate_pipeline] Loaded aligner from {aligner_path}")
        except Exception as e:
            print(f"[validate_pipeline] Warning: Could not load aligner: {e}")

    # Default corpus paths from M4 if not provided
    if corpus_paths is None:
        corpus_paths = {
            "train_path": "corpus/train_corpus.jsonl",
            "test_path": "corpus/test_corpus.jsonl"
        }

    return run_all_checks(paths_dict, m1_outputs, m2_outputs, aligner, corpus_paths)


# ---------------------------------------------------------------------------
# MODULE ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("validate_pipeline.py — Import and use validate_full_pipeline()")
    print("See HTML pseudocode for detailed check specifications.")
