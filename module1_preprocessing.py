# =======================================================================
# scMultiPert — Module 1: Data Ingestion & Preprocessing
# =======================================================================
# 
# This module orchestrates the ingestion and standardization of multi-modal 
# datasets from the Perturbation Catalogue, ensuring compatibility with 
# downstream foundational models like Geneformer.
#
# Data Modalities:
#   - scPerturb-seq  : Norman et al. 2019 (CRISPRa screen, K562 cells)
#   - CRISPR screens : DepMap 24Q2 gene effect scores
#   - MAVE (Multi)   : BRCA1, TP53, and PTEN saturation genome editing
#
# Architecture & Standards:
#   - Enforces AnnData integrity with explicit type hinting and proper 
#     import aliasing (anndata as ad).
#   - Implements full modality loading for CRISPR and MAVE to prevent
#     AttributeErrors in subsequent processing stages.
#   - Standardizes metadata schemas to meet Module 2 validation contracts,
#     specifically ensuring 'perturbed_genes' availability in cell obs.
#
# Output Contract (Consumed by Modules 2, 4, and 6):
#   {
#     "scrna"  : AnnData object
#                - obs: ['perturbation', 'is_control', 'perturbed_genes']
#                - layers["counts"]: Raw integer counts (Geneformer req.)
#                - var_names: Ensembl IDs (Geneformer req.)
#
#     "crispr" : DataFrame
#                - columns: ['gene', 'lineage', 'fitness_score', 
#                            'is_essential', 'cell_line_id']
#
#     "mave"   : DataFrame
#                - columns: ['variant_id', 'gene_symbol', 'position', 
#                            'aa_ref', 'aa_alt', 'sge_score', 
#                            'function_class', 'wt_sequence', 'mutant_sequence']
#   }
#
# Dependencies: scanpy, anndata, pandas, numpy, geneformer
# =======================================================================

import scanpy as sc
import anndata as ad          
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# SECTION 0 — Gene metadata & custom exceptions
# ---------------------------------------------------------------------------

# Reference lengths for well-known MAVE genes.
# _process_single_mave() checks the config dict first, then falls back here.
# Genes absent from both emit a warning and proceed.

_GENE_METADATA = {
    "BRCA1": {"expected_length": 1863, "uniprot": "P38398"},
    "TP53":  {"expected_length": 393,  "uniprot": "P04637"},
    "PTEN":  {"expected_length": 403,  "uniprot": "P60484"},
}


class Module1Error(Exception):
    """Base class for Module 1 errors."""
    pass


class FileNotFoundError_(Module1Error):
    """File missing or unreadable."""
    pass


class MissingColumnError(Module1Error):
    """Required column absent."""
    pass


class EmptyDataError(Module1Error):
    """Filter results in empty dataset."""
    pass


class SequenceError(Module1Error):
    """Invalid protein sequence."""
    pass


class GeneVocabError(Module1Error):
    """Incompatible with Geneformer vocab."""
    pass


# ---------------------------------------------------------------------------
# SHARED HELPERS
# ---------------------------------------------------------------------------

def _assert_file_exists(path: str, label: str) -> None:
    """Verify that a file exists and is readable."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError_(f"[Module 1] {label} not found at: '{path}'")


def _assert_columns_present(df: pd.DataFrame, required: list, source: str) -> None:
    """Verify that all required columns are present in a DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnError(
            f"[Module 1] {source} missing columns: {missing}"
        )


def _assert_nonempty(n: int, label: str) -> None:
    """Verify that a dataset is not empty after processing."""
    if n == 0:
        raise EmptyDataError(f"[Module 1] {label} is empty after processing.")


# ---------------------------------------------------------------------------
# SECTION 1 — scPerturb-seq (Norman et al. 2019)
# ---------------------------------------------------------------------------

def load_norman2019(
    h5ad_path:   str,
    min_genes:   int   = 200,
    max_mt_pct:  float = 20.0,
    min_cells:   int   = 3,
) -> ad.AnnData:
    """
    Load and standardize Norman 2019 CRISPRa screen data for Geneformer compatibility.
    
    Processing Pipeline:
      1. ID Mapping: Symbol to Ensembl conversion (required for Geneformer tokens).
      2. Quality Control: MT-gene filtering and cell/gene sparsity pruning.
      3. Data Preservation: Stores raw counts in layers["counts"] prior to normalization.
      4. Metadata Assignment: Defines 'is_control' and 'perturbed_genes' in obs.
    
    Args:
        h5ad_path: Path to the Norman 2019 .h5ad source file.
        min_genes: Minimum genes per cell (QC threshold).
        max_mt_pct: Maximum mitochondrial read percentage (QC threshold).
        min_cells: Minimum cells per gene (QC threshold).
    
    Returns:
        AnnData: Standardized object with:
          - obs: ['perturbation', 'is_control', 'perturbed_genes']
          - layers: ['counts'] (raw integer counts)
          - var_names: Ensembl IDs
    """
    _assert_file_exists(h5ad_path, "Norman 2019 .h5ad")
    adata = sc.read_h5ad(h5ad_path)
    print(f"[Norman 2019] Raw: {adata.n_obs} cells | {adata.n_vars} genes")

    # Step 1 — Geneformer requires Ensembl IDs as var_names
    adata = _ensure_ensembl_var_names(adata)

    # Step 2 — QC metrics
    # After Ensembl conversion, MT genes are identified by their stored symbol.
    if "gene_symbol" in adata.var.columns:
        adata.var["mt"] = adata.var["gene_symbol"].str.startswith("MT-")
    else:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["pct_counts_mt"] < max_mt_pct].copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Step 3 — Preserve raw counts before normalisation.
    # Geneformer tokeniser ranks genes by expression and requires integer
    # counts from layers["counts"], not log-normalised values.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Step 4 — Perturbation metadata
    # is_control: boolean flag consumed by Module 4 centroid computation.
    adata.obs["is_control"] = adata.obs["perturbation"] == "control"

    # perturbed_genes — list of atomic gene names per cell.
    # Module 2 declares this as a required obs column and includes it in 
    # cell_metadata. Controls get an empty list.
    # Combinatorial perturbations (e.g. "CDKN1A+FOXA1") are split on "+".
    adata.obs["perturbed_genes"] = adata.obs["perturbation"].apply(
        lambda x: [] if x == "control" else x.split("+")
    )

    _assert_nonempty(adata.n_obs, "Norman 2019 cells after QC")
    print(f"[Norman 2019] Final: {adata.n_obs} cells | {adata.n_vars} genes")
    return adata


# ---------------------------------------------------------------------------
# SECTION 2 — CRISPR screens (DepMap 24Q2)
# ---------------------------------------------------------------------------

def load_depmap_crispr(
    gene_effect_path:  str,
    sample_info_path:  str,
    fitness_threshold: float = -0.5,
    min_cell_lines:    int   = 5,
) -> pd.DataFrame:
    """
    Load and process DepMap 24Q2 Chronos CRISPR screen data.
    
    Ingests gene effect scores to identify essential genes across diverse 
    cancer cell lines. Ensures continuous data flow to Module 2.
    
    Processing steps:
      - Loads wide-format gene effect matrix
      - Transforms to long-format (one row per cell_line/gene pair)
      - Attaches lineage metadata from sample_info
      - Classifies gene essentiality based on Chronos scores
      - Retains genes essential in at least min_cell_lines lines
    
    Args:
        gene_effect_path: Path to CRISPRGeneEffect.csv (Cell lines x Genes).
        sample_info_path: Path to sample_info.csv (Metadata index).
        fitness_threshold: Chronos score threshold for essentiality.
        min_cell_lines: Minimum cell lines required for gene retention.
    
    Returns:
        DataFrame in long format with columns:
          - cell_line_id, cell_line_name, lineage, lineage_subtype
          - gene, fitness_score, is_essential
    """
    _assert_file_exists(gene_effect_path, "DepMap gene effect CSV")
    _assert_file_exists(sample_info_path, "DepMap sample info CSV")

    # Load gene effect matrix — columns are "SYMBOL (ENTREZ_ID)", extract symbol
    gene_effect = pd.read_csv(gene_effect_path, index_col=0)
    gene_effect.columns = gene_effect.columns.str.extract(r"^([^\s]+)")[0]

    sample_info = pd.read_csv(sample_info_path, index_col=0)

    # Melt to long format: one row per (cell_line, gene) pair
    df_long = (
        gene_effect.reset_index()
        .melt(id_vars="index", var_name="gene", value_name="fitness_score")
        .rename(columns={"index": "cell_line_id"})
        .dropna(subset=["fitness_score"])
    )

    # Attach lineage metadata
    df_long = df_long.merge(
        sample_info[["cell_line_name", "lineage", "lineage_subtype"]],
        left_on="cell_line_id",
        right_index=True,
        how="left",
    )

    # Essentiality flag
    df_long["is_essential"] = df_long["fitness_score"] < fitness_threshold

    # Retain only genes that are essential in at least min_cell_lines lines
    ess_counts = (
        df_long[df_long["is_essential"]]
        .groupby("gene")["cell_line_id"]
        .nunique()
    )
    valid_genes = ess_counts[ess_counts >= min_cell_lines].index
    df_long = df_long[df_long["gene"].isin(valid_genes)].copy()

    _assert_nonempty(len(df_long), "DepMap rows after filtering")
    print(
        f"[DepMap 24Q2] Retained {len(valid_genes)} essential genes "
        f"| {len(df_long)} rows"
    )
    return df_long


# ---------------------------------------------------------------------------
# SECTION 3 — MAVE (multi-gene: BRCA1, TP53, PTEN)
# ---------------------------------------------------------------------------

def load_multi_mave(mave_configs: list) -> pd.DataFrame:
    """
    Aggregate and harmonize MAVE datasets into unified functional genomics data.
    
    Consolidates diverse MAVE datasets (BRCA1, TP53, PTEN) into a canonical 
    schema enabling direct comparison of mutational effects across assays.
    
    Args:
        mave_configs: List of dictionaries containing:
            - gene_symbol: Canonical symbol (e.g., "BRCA1", "TP53").
            - csv_path: Path to variant data CSV.
            - fasta_path: Path to reference sequence FASTA.
            - column_map: Mapping of raw headers to standardized names.
            - min_coverage: Threshold for experimental read depth.
            - expected_length: Sequence length validation (overrides defaults).
    
    Returns:
        DataFrame with unified schema:
          - variant_id, gene_symbol, position, aa_ref, aa_alt
          - sge_score, function_class, wt_sequence, mutant_sequence
    """
    all_dfs = []
    for cfg in mave_configs:
        gene = cfg["gene_symbol"]
        print(f"[MAVE] Processing {gene}...")
        all_dfs.append(_process_single_mave(cfg))

    combined = pd.concat(all_dfs, ignore_index=True)
    _assert_nonempty(len(combined), "Combined MAVE datasets")
    print(
        f"[MAVE] Combined: {len(combined)} variants across "
        f"{combined['gene_symbol'].unique().tolist()}"
    )
    return combined


def _process_single_mave(cfg: dict) -> pd.DataFrame:
    """
    Process one MAVE CSV / FASTA pair into the canonical format.
    
    Sequence validation priority:
      1. cfg["expected_length"] (caller override)
      2. _GENE_METADATA[gene]["expected_length"] (known genes)
      3. None → warning only, proceed with FASTA length as-is
    
    Args:
        cfg: Configuration dictionary for a single gene MAVE dataset.
    
    Returns:
        DataFrame with standardized columns for variant analysis.
    """
    gene = cfg["gene_symbol"]
    _assert_file_exists(cfg["csv_path"], f"{gene} CSV")
    _assert_file_exists(cfg["fasta_path"], f"{gene} FASTA")

    df = pd.read_csv(cfg["csv_path"]).rename(columns=cfg["column_map"])

    # Require variant_id and sge_score after column renaming
    _assert_columns_present(df, ["variant_id", "sge_score"], f"MAVE {gene}")

    # Optional coverage filter
    if "coverage" in df.columns and "min_coverage" in cfg:
        df = df[df["coverage"] >= cfg["min_coverage"]].copy()

    # HGVS parsing — supports both 3-letter (p.Arg123His) and 1-letter (R123H)
    parsed = df["variant_id"].str.extract(r"([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})")
    df[["aa_ref_raw", "position", "aa_alt_raw"]] = parsed.values
    df = df.dropna(subset=["position"]).copy()
    df["position"] = df["position"].astype(int)

    # Standardise to single-letter amino acid codes
    aa_map = _get_aa_mapping()
    def _to_1(x):
        return aa_map.get(x, x) if len(x) > 1 else x
    df["aa_ref"] = df["aa_ref_raw"].apply(_to_1)
    df["aa_alt"] = df["aa_alt_raw"].apply(_to_1)

    # Sequence validation
    wt_seq = _read_fasta(cfg["fasta_path"])
    expected = (
        cfg.get("expected_length")
        or _GENE_METADATA.get(gene, {}).get("expected_length")
    )
    if expected is not None:
        if abs(len(wt_seq) - expected) > 10:
            raise SequenceError(
                f"[Module 1] {gene} FASTA length {len(wt_seq)} "
                f"differs from expected {expected} by more than 10 residues."
            )
    else:
        print(
            f"[Module 1] Warning: no reference length for '{gene}'. "
            f"Proceeding with FASTA length {len(wt_seq)}."
        )

    # Generate mutant sequences
    df["wt_sequence"] = wt_seq
    df["gene_symbol"] = gene
    df["mutant_sequence"] = df.apply(
        lambda r: _apply_substitution(wt_seq, r["position"], r["aa_alt"]),
        axis=1,
    )
    df = df.dropna(subset=["mutant_sequence"]).copy()

    # Functional classification fallback
    if "function_class" not in df.columns:
        df["function_class"] = pd.cut(
            df["sge_score"],
            bins=[-np.inf, -1.0, -0.5, np.inf],
            labels=["loss_of_function", "intermediate", "functional"],
        )

    # Prefix variant IDs with gene symbol to avoid cross-gene collisions
    # in Module 4/5 lookup dicts (e.g. "BRCA1_p.Arg1699Trp")
    df["variant_id"] = gene + "_" + df["variant_id"].astype(str)

    return df[[
        "variant_id", "gene_symbol", "position", "aa_ref", "aa_alt",
        "sge_score", "function_class", "wt_sequence", "mutant_sequence",
    ]]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _read_fasta(path: str) -> str:
    """Read a FASTA file and return the sequence as an uppercase string."""
    with open(path, "r") as f:
        return "".join(
            line.strip() for line in f if not line.startswith(">")
        ).upper()


def _apply_substitution(seq: str, pos: int, alt: str):
    """
    Apply a single amino acid substitution at 1-based position.
    
    Returns None for stop codons (*) or out-of-bounds positions so that
    rows with invalid substitutions can be dropped cleanly downstream.
    """
    idx = pos - 1
    if idx < 0 or idx >= len(seq) or alt == "*":
        return None
    return seq[:idx] + alt + seq[idx + 1:]


def _get_aa_mapping() -> dict:
    """Three-letter to one-letter amino acid code mapping."""
    return {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        "Ter": "*",
    }


def _ensure_ensembl_var_names(adata: ad.AnnData) -> ad.AnnData:
    """
    Convert gene symbols in adata.var_names to Ensembl IDs for Geneformer.
    
    If more than half of the first 100 var_names already look like Ensembl
    IDs (start with "ENSG"), the AnnData is returned as-is to avoid
    double-mapping. Genes without a mapping are dropped and a count is
    printed so the caller knows how many were lost.
    
    Raises:
        GeneVocabError: Geneformer mapping dict not found on disk.
    """
    # Guard: already Ensembl?
    sample = list(adata.var_names[:100])
    if sum(1 for g in sample if str(g).startswith("ENSG")) > 50:
        if "gene_symbol" not in adata.var.columns:
            adata.var["gene_symbol"] = list(adata.var_names)
        return adata

    import pickle
    import geneformer
    dict_path = Path(geneformer.__file__).parent / "gene_name_id_dict.pkl"
    if not dict_path.exists():
        raise GeneVocabError(
            "[Module 1] Geneformer mapping dict not found. "
            "Reinstall Geneformer from HuggingFace."
        )

    with open(dict_path, "rb") as f:
        sym_to_ens = pickle.load(f)

    mapped = [sym_to_ens.get(s, None) for s in adata.var_names]
    keep = [m is not None for m in mapped]

    n_dropped = sum(1 for m in mapped if m is None)
    new_adata = adata[:, keep].copy()
    new_adata.var["gene_symbol"] = list(new_adata.var_names)
    new_adata.var_names = [m for m in mapped if m is not None]

    print(
        f"[Geneformer] Mapped {new_adata.n_vars} genes to Ensembl IDs "
        f"({n_dropped} symbols had no mapping and were dropped)."
    )
    return new_adata


# ---------------------------------------------------------------------------
# UNIFIED LOADER
# ---------------------------------------------------------------------------

def load_all_modalities(
    norman_path:        str,
    depmap_effect_path: str,
    depmap_info_path:   str,
    mave_configs:       list,
) -> dict:
    """
    Module 1 Entry Point: Multi-Modal Data Orchestration.
    
    Coordinates the simultaneous loading and standardization of scRNA-seq, 
    CRISPR, and MAVE data modalities. Ensures all outputs adhere to the 
    strict data contracts required by downstream processing modules.
    
    Args:
        norman_path: Path to Norman 2019 .h5ad source file.
        depmap_effect_path: Path to DepMap CRISPRGeneEffect.csv.
        depmap_info_path: Path to DepMap sample_info.csv.
        mave_configs: List of per-gene configuration dictionaries.
    
    Returns:
        Dictionary containing:
          - "scrna": AnnData [Ensembl IDs, raw counts, perturbation metadata].
          - "crispr": DataFrame [Essentiality scores, lineage, cell line IDs].
          - "mave": DataFrame [Variant-to-sequence mappings, SGE scores].
    
    Note:
        This unified output is the primary dependency for Module 2 (Encoding), 
        Module 4 (Training), and Module 6 (Inference).
    """
    print("=" * 55)
    print("scMultiPert | Module 1: Preprocessing (v1.0-proposal)")
    print("=" * 55)

    scrna_data = load_norman2019(norman_path)
    crispr_data = load_depmap_crispr(depmap_effect_path, depmap_info_path)
    mave_data = load_multi_mave(mave_configs)

    # Sanity check: Geneformer requires raw counts in layers
    if "counts" not in scrna_data.layers:
        raise Module1Error(
            "[Module 1] scrna.layers['counts'] is missing after load_norman2019(). "
            "This is required by Module 2's Geneformer tokeniser."
        )

    print("\n[Module 1] Preprocessing complete.")
    print(f"  scRNA  : {scrna_data.n_obs} cells | {scrna_data.n_vars} genes")
    print(f"  CRISPR : {crispr_data['gene'].nunique()} essential genes")
    print(
        f"  MAVE   : {len(mave_data)} variants across "
        f"{mave_data['gene_symbol'].unique().tolist()}"
    )

    return {
        "scrna": scrna_data,
        "crispr": crispr_data,
        "mave": mave_data,
    }