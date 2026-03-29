# =======================================================================
# scMultiPert — Module 6: Benchmarking & Evaluation (v1.0-proposal)
# =======================================================================
#
# This module executes comprehensive benchmarking across three evaluation
# axes: MAVE (variant effect prediction), scRNA-seq (transcriptional
# response generation), and generalist capability retention. It implements
# ProteinGym-style and GEARS-style metrics while maintaining full
# compatibility with upstream module outputs.
#
# Architecture Integration:
#   - Module 2 (Encoders): Consumes var_embeddings [N, 320], centroids dict,
#     cell_metadata DataFrame, var_texts, var_ids, crispr_texts.
#   - Module 3 (Alignment): Loads BiomedicalAligner for soft token generation
#     and attention extraction.
#   - Module 4 (Corpus): Consumes test_corpus.jsonl with ChatMLExample format,
#     gene-based split metadata, and perturbation centroids.
#   - Module 5 (Fine-tuning): Loads saved LoRA adapters and aligner checkpoints.
#
# Baseline Strategy:
#   - Baseline A (BM25): Lexical retrieval over var_texts for MAVE ranking.
#   - Baseline B (MLP): Pure numeric regression (cell_emb ⊕ var_emb → score).
#   - Baseline C (BioMistral-7B): Zero-shot without LoRA adapters.
#   - Baseline D (Generalist): Llama-3 8B or GPT-4o via API for gap analysis.
#
# Dependencies: torch, transformers, peft, scipy, scikit-learn, pandas,
#               rank_bm25, tqdm
# =======================================================================

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
import json

# Module imports for interface compatibility
from module2_encoders import ESMVariantEncoder, GeneformerEncoder
from module3_alignment import BiomedicalAligner, load_aligner
from module4_corpus import ChatMLExample, BIO_CONTEXT_TOKEN, _extract_gene_components
from module5_finetuning import load_biomistral_4bit, get_qlora_config, BIOMISTRAL_MODEL

# ---------------------------------------------------------------------------
# SECTION 0 — Custom Exceptions & Data Structures
# ---------------------------------------------------------------------------

class Module6Error(Exception):
    """Base class for Module 6 benchmarking errors."""
    pass


class CheckpointError(Module6Error):
    """Raised when model checkpoint loading fails."""
    pass


class EvaluationError(Module6Error):
    """Raised when metric computation encounters invalid data."""
    pass


@dataclass
class BenchmarkResult:
    """
    Structured container for per-model, per-task benchmark metrics.
    
    Attributes:
        model_name: Identifier string (e.g., "scMultiPert", "BM25", "MLP").
        task: Evaluation task ("mave", "scrna", "medqa", "pubmedqa").
        metric_name: Specific metric (e.g., "per_protein_spearman", "gene_set_f1").
        metric_value: Float result (or dict for per-gene breakdowns).
        metadata: Optional dict with per-gene, per-sample, or config details.
    """
    model_name: str
    task: str
    metric_name: str
    metric_value: float
    metadata: Optional[Dict] = None


@dataclass 
class MAVEPrediction:
    """
    Prediction container for MAVE variant effect scoring.
    
    Required for per_protein_spearman() and top_k_recall() compatibility
    with Module 4's mave_df schema.
    """
    variant_id: str           # Matches mave_df["variant_id"] format: "GENE_p.Aaa123Bbb"
    gene_symbol: str          # Matches mave_df["gene_symbol"]
    predicted_score: float    # Model-predicted functional score (continuous)
    true_score: float         # Ground truth sge_score from mave_df
    predicted_class: str      # "loss_of_function", "intermediate", "functional"
    true_class: str           # Ground truth function_class from mave_df


# ---------------------------------------------------------------------------
# SECTION 1 — Entry Point & Orchestration
# ---------------------------------------------------------------------------

def run_benchmarking(
    checkpoints: Dict[str, str],
    m2_outputs: Dict[str, any],
    m4_data: Dict[str, any],
    m1_modalities: Dict[str, any],
    output_dir: str = "benchmark_results/"
) -> pd.DataFrame:
    """
    Main entry point: Execute complete benchmark suite across all baselines.
    
    Args:
        checkpoints: Dict mapping model identifiers to checkpoint paths.
            Required keys: "scmultipert", "aligner", "biomistral_base"
            Example: {
                "scmultipert": "finetuned_stage2/llm_adapter/",
                "aligner": "finetuned_stage2/aligner_stage2_final.pt",
                "biomistral_base": None  # Load from HF hub, no adapter
            }
        m2_outputs: Module 2 output dictionary containing:
            - "cell_embeddings": Tensor [N_cells, 256]
            - "cell_metadata": DataFrame with embedding_label, is_control
            - "var_embeddings": Tensor [N_valid_variants, 320]
            - "var_texts": List[str] aligned with var_ids
            - "var_ids": List[str] aligned with var_embeddings
            - "crispr_texts": List[str]
            - "crispr_agg": DataFrame
        m4_data: Module 4 output dictionary containing:
            - "train_path": Path to train_corpus.jsonl
            - "test_path": Path to test_corpus.jsonl
            - "centroids": Dict[str, Tensor] perturbation centroids
            - "gene_vocabulary": Set[str] all genes
        m1_modalities: Module 1 output dictionary for raw data access:
            - "scrna": AnnData with perturbation metadata
            - "mave": DataFrame with variant annotations
            - "crispr": DataFrame with gene effect scores
    
    Returns:
        DataFrame with columns [model_name, task, metric_name, metric_value, metadata]
        containing all benchmark results for analysis and visualization.
    
    Raises:
        CheckpointError: If required checkpoints cannot be loaded.
        EvaluationError: If test data is incompatible with evaluation logic.
    """
    print("=" * 60)
    print("scMultiPert | Module 6: Benchmarking Suite (v1.0-proposal)")
    print("=" * 60)
    
    # Validate input contracts
    _validate_benchmark_inputs(checkpoints, m2_outputs, m4_data)
    
    # Load test corpus from Module 4 output
    test_examples = _load_test_corpus(m4_data["test_path"])
    
    results: List[BenchmarkResult] = []
    
    # ======================================================================
    # SECTION 1.1 — MAVE Benchmark (ProteinGym-style)
    # ======================================================================
    print("\n[Benchmark] Running MAVE Variant Effect Prediction...")
    
    # Load scMultiPert (full system with LoRA + Aligner)
    scmp_model = load_scmultipert(checkpoints["scmultipert"], checkpoints["aligner"])
    
    mave_results = evaluate_mave(
        model=scmp_model,
        test_examples=test_examples,
        mave_df=m1_modalities["mave"],
        var_embeddings=m2_outputs["var_embeddings"],
        var_ids=m2_outputs["var_ids"],
        centroids=m4_data["centroids"]
    )
    results.extend(_tag_results(mave_results, "scMultiPert", "mave"))
    
    # Baseline A: BM25 Lexical Retrieval
    print("\n[Baseline A] BM25 Lexical Retrieval...")
    bm25_model = load_bm25_predictor(m2_outputs["var_texts"], m2_outputs["var_ids"])
    bm25_results = evaluate_mave(
        model=bm25_model,
        test_examples=test_examples,
        mave_df=m1_modalities["mave"],
        var_embeddings=None,  # BM25 doesn't use embeddings
        var_ids=m2_outputs["var_ids"],
        centroids=None
    )
    results.extend(_tag_results(bm25_results, "BM25", "mave"))
    
    # Baseline B: MLP Numeric Regressor
    print("\n[Baseline B] MLP Pure-Numeric Regressor...")
    mlp_model = load_mlp_regressor(
        embeddings=m2_outputs["var_embeddings"],
        labels=_extract_mave_labels(m1_modalities["mave"], m2_outputs["var_ids"])
    )
    mlp_results = evaluate_mave(
        model=mlp_model,
        test_examples=test_examples,
        mave_df=m1_modalities["mave"],
        var_embeddings=m2_outputs["var_embeddings"],
        var_ids=m2_outputs["var_ids"],
        centroids=m4_data["centroids"]
    )
    results.extend(_tag_results(mlp_results, "MLP-Numeric", "mave"))
    
    # Baseline C: BioMistral Zero-shot (no LoRA)
    print("\n[Baseline C] BioMistral-7B Zero-shot...")
    biomistral_zs = load_biomistral_zeroshot()
    bm_zs_results = evaluate_mave(
        model=biomistral_zs,
        test_examples=test_examples,
        mave_df=m1_modalities["mave"],
        var_embeddings=None,
        var_ids=m2_outputs["var_ids"],
        centroids=None
    )
    results.extend(_tag_results(bm_zs_results, "BioMistral-ZeroShot", "mave"))
    
    # Baseline D: Generalist LLM (Llama-3 or GPT-4o)
    if "generalist" in checkpoints:
        print("\n[Baseline D] Generalist LLM (Llama-3/GPT-4o)...")
        generalist = load_generalist_llm(checkpoints["generalist"])
        gen_results = evaluate_mave(
            model=generalist,
            test_examples=test_examples,
            mave_df=m1_modalities["mave"],
            var_embeddings=None,
            var_ids=m2_outputs["var_ids"],
            centroids=None
        )
        results.extend(_tag_results(gen_results, "Generalist-LLM", "mave"))
    
    # ======================================================================
    # SECTION 1.2 — scRNA Benchmark (GEARS-style)
    # ======================================================================
    print("\n[Benchmark] Running scRNA-seq Response Prediction...")
    
    scrna_results = evaluate_scrna(
        model=scmp_model,
        test_examples=test_examples,
        centroids=m4_data["centroids"],
        adata=m1_modalities["scrna"]
    )
    results.extend(_tag_results(scrna_results, "scMultiPert", "scrna"))
    
    # ID vs OOD Gap Analysis
    id_ood_gap = calculate_id_ood_gap(scrna_results, m4_data["gene_vocabulary"])
    results.append(BenchmarkResult(
        model_name="scMultiPert",
        task="scrna",
        metric_name="id_vs_ood_gap",
        metric_value=id_ood_gap,
        metadata={"description": "F1 difference between seen and unseen genes"}
    ))
    
    # ======================================================================
    # SECTION 1.3 — Generalist Capability Retention
    # ======================================================================
    print("\n[Benchmark] Running Generalist Capability Checks...")
    
    # MedQA-USMLE subset evaluation
    medqa_acc = evaluate_medqa(scmp_model, biomistral_zs)
    results.append(BenchmarkResult(
        model_name="scMultiPert",
        task="medqa",
        metric_name="accuracy",
        metric_value=medqa_acc["scmp"],
        metadata={"baseline_accuracy": medqa_acc["baseline"]}
    ))
    
    # PubMedQA evaluation
    pubmedqa_acc = evaluate_pubmedqa(scmp_model, biomistral_zs)
    results.append(BenchmarkResult(
        model_name="scMultiPert",
        task="pubmedqa",
        metric_name="accuracy",
        metric_value=pubmedqa_acc["scmp"],
        metadata={"baseline_accuracy": pubmedqa_acc["baseline"]}
    ))
    
    # ======================================================================
    # SECTION 1.4 — Interpretability Analysis
    # ======================================================================
    print("\n[Benchmark] Running Interpretability Extraction...")
    
    # Attention heatmap extraction from Aligner
    attn_results = extract_attention_heatmaps(
        aligner=scmp_model["aligner"],
        test_batch=_sample_test_batch(test_examples, n=100)
    )
    results.append(BenchmarkResult(
        model_name="scMultiPert",
        task="interpretability",
        metric_name="attention_heatmap_samples",
        metric_value=len(attn_results),
        metadata={"sample_ids": list(attn_results.keys())}
    ))
    
    # LOF classification from soft tokens only (no text)
    lof_f1 = lof_softtoken_classification(
        aligner=scmp_model["aligner"],
        centroids=m4_data["centroids"],
        var_embeddings=m2_outputs["var_embeddings"],
        mave_df=m1_modalities["mave"]
    )
    results.append(BenchmarkResult(
        model_name="scMultiPert",
        task="interpretability",
        metric_name="lof_softtoken_f1",
        metric_value=lof_f1,
        metadata={"description": "F1 classifying LOF from soft tokens without text"}
    ))
    
    # Convert to DataFrame and save
    results_df = _results_to_dataframe(results)
    _save_benchmark_results(results_df, output_dir)
    
    print("\n" + "=" * 60)
    print("Module 6 Complete")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return results_df


# ---------------------------------------------------------------------------
# SECTION 2 — Model Loading Functions
# ---------------------------------------------------------------------------

def load_scmultipert(
    lora_path: str,
    aligner_path: str,
    device: torch.device = None
) -> Dict[str, any]:
    """
    Load complete scMultiPert system: BioMistral + LoRA + Aligner.
    
    Args:
        lora_path: Directory containing LoRA adapter weights (Module 5 output).
        aligner_path: Path to BiomedicalAligner checkpoint (Module 5 output).
        device: Target compute device (auto-detected if None).
    
    Returns:
        Dictionary with keys:
            - "llm": PeftModel (BioMistral + LoRA)
            - "tokenizer": AutoTokenizer
            - "aligner": BiomedicalAligner (unfrozen, eval mode)
            - "is_scmultipert": True (flag for evaluation routing)
    
    Raises:
        CheckpointError: If LoRA or aligner checkpoints fail to load.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer (consistent with Module 5)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with 4-bit quantization (Module 5 config)
    base_model = load_biomistral_4bit()
    
    # Load LoRA adapters (Module 5 output)
    try:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(base_model, lora_path)
        llm.eval().to(device)
    except Exception as e:
        raise CheckpointError(f"Failed to load LoRA adapters from {lora_path}: {e}")
    
    # Load trained aligner (Module 3/5 output)
    try:
        aligner = load_aligner(aligner_path)
        aligner.to(device)
        aligner.eval()
    except Exception as e:
        raise CheckpointError(f"Failed to load aligner from {aligner_path}: {e}")
    
    print(f"[load_scmultipert] Loaded complete system on {device}")
    
    return {
        "llm": llm,
        "tokenizer": tokenizer,
        "aligner": aligner,
        "is_scmultipert": True,
        "device": device
    }


def load_biomistral_zeroshot(device: torch.device = None) -> Dict[str, any]:
    """
    Load base BioMistral-7B without LoRA adapters for zero-shot baseline.
    
    Args:
        device: Target compute device.
    
    Returns:
        Dictionary with same interface as load_scmultipert but without
        aligner and LoRA components. Evaluation code routes to text-only
        generation without soft token injection.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model without quantization for fair comparison (or use 4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        BIOMISTRAL_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    return {
        "llm": model,
        "tokenizer": tokenizer,
        "aligner": None,  # No aligner in zero-shot
        "is_scmultipert": False,
        "is_zeroshot": True,
        "device": device
    }


def load_bm25_predictor(
    var_texts: List[str],
    var_ids: List[str]
) -> Dict[str, any]:
    """
    Initialize BM25 lexical retrieval baseline for MAVE ranking.
    
    Uses rank_bm25 to index variant text descriptions for lexical matching.
    Compatible with Module 2's var_texts/var_ids output format.
    
    Args:
        var_texts: List of structured variant descriptions from Module 2.
        var_ids: Corresponding variant identifiers.
    
    Returns:
        Dictionary with BM25 index and metadata for evaluate_mave routing.
    """
    from rank_bm25 import BM25Okapi
    
    # Tokenize texts for BM25 (simple whitespace tokenization)
    tokenized_corpus = [text.lower().split() for text in var_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return {
        "bm25": bm25,
        "var_texts": var_texts,
        "var_ids": var_ids,
        "tokenized_corpus": tokenized_corpus,
        "is_bm25": True
    }


def load_mlp_regressor(
    embeddings: torch.Tensor,
    labels: pd.Series,
    hidden_dim: int = 256,
    epochs: int = 50
) -> Dict[str, any]:
    """
    Train MLP baseline: Direct numeric regression from embeddings to scores.
    
    Concatenates cell and variant embeddings where applicable, or uses
    variant embeddings alone for MAVE score prediction.
    
    Args:
        embeddings: Tensor [N, D] from Module 2 (var_embeddings or concat).
        labels: Series of ground truth scores aligned with embeddings.
        hidden_dim: MLP hidden layer size.
        epochs: Training iterations.
    
    Returns:
        Dictionary with trained MLP model and preprocessing metadata.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Convert to numpy
    X = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    y = labels.values if isinstance(labels, pd.Series) else labels
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
        activation='relu',
        solver='adam',
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_scaled, y)
    
    return {
        "mlp": mlp,
        "scaler": scaler,
        "is_mlp": True,
        "input_dim": X.shape[1]
    }


def load_generalist_llm(
    model_path_or_api_key: str,
    model_type: str = "llama3"
) -> Dict[str, any]:
    """
    Load generalist LLM baseline (Llama-3 8B local or GPT-4o via API).
    
    Args:
        model_path_or_api_key: Local path for Llama-3 or API key for GPT-4o.
        model_type: "llama3" or "gpt4o".
    
    Returns:
        Dictionary with model interface compatible with evaluate_mave.
    """
    if model_type == "llama3":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_api_key)
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_api_key,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        return {
            "llm": model,
            "tokenizer": tokenizer,
            "is_generalist": True,
            "model_type": "llama3"
        }
    
    elif model_type == "gpt4o":
        import openai
        
        return {
            "client": openai.OpenAI(api_key=model_path_or_api_key),
            "model_name": "gpt-4o",
            "is_generalist": True,
            "model_type": "gpt4o"
        }
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ---------------------------------------------------------------------------
# SECTION 3 — MAVE Benchmark (ProteinGym-style)
# ---------------------------------------------------------------------------

def evaluate_mave(
    model: Dict[str, any],
    test_examples: List[ChatMLExample],
    mave_df: pd.DataFrame,
    var_embeddings: Optional[torch.Tensor],
    var_ids: List[str],
    centroids: Optional[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Execute ProteinGym-style evaluation on MAVE variant effect prediction.
    
    Routes to appropriate evaluation logic based on model type:
        - scMultiPert: Soft token injection via aligner
        - BM25: Lexical retrieval and ranking
        - MLP: Direct embedding regression
        - Zero-shot LLMs: Text-only generation
    
    Args:
        model: Model dictionary from load_* functions.
        test_examples: Filtered ChatMLExample list with modality=="mave".
        mave_df: Module 1 MAVE DataFrame with ground truth annotations.
        var_embeddings: Module 2 variant embeddings (for numeric baselines).
        var_ids: Module 2 variant ID list aligned with embeddings.
        centroids: Module 4 perturbation centroids (for cell context).
    
    Returns:
        Dictionary of metric_name -> metric_value:
            - "per_protein_spearman": Dict[str, float] (per-gene ρ)
            - "mean_spearman": Float (average across genes)
            - "top_k_recall_0.10": Float (recall of top 10% LOF variants)
            - "directionality_accuracy": Float (beneficial vs detrimental)
    """
    # Filter to MAVE examples only
    mave_examples = [ex for ex in test_examples if ex.modality == "mave"]
    
    if len(mave_examples) == 0:
        raise EvaluationError("No MAVE examples found in test corpus")
    
    # Route to appropriate evaluation strategy
    if model.get("is_bm25"):
        predictions = _evaluate_mave_bm25(model, mave_examples, mave_df)
    elif model.get("is_mlp"):
        predictions = _evaluate_mave_mlp(model, mave_examples, mave_df, var_embeddings, var_ids)
    elif model.get("is_scmultipert"):
        predictions = _evaluate_mave_scmultipert(model, mave_examples, mave_df, centroids)
    else:
        predictions = _evaluate_mave_llm(model, mave_examples, mave_df)
    
    # Compute metrics
    results = {}
    
    # Per-protein Spearman correlation
    results["per_protein_spearman"] = per_protein_spearman(predictions, mave_df)
    results["mean_spearman"] = np.mean(list(results["per_protein_spearman"].values()))
    
    # Top-K recall (K=10% most harmful variants)
    results["top_k_recall_0.10"] = top_k_recall(predictions, k=0.10)
    
    # Directionality classification accuracy
    results["directionality_accuracy"] = directionality_check(predictions)
    
    return results


def _evaluate_mave_scmultipert(
    model: Dict[str, any],
    examples: List[ChatMLExample],
    mave_df: pd.DataFrame,
    centroids: Dict[str, torch.Tensor]
) -> List[MAVEPrediction]:
    """
    Internal: scMultiPert evaluation with soft token injection.
    
    Uses Module 3 BiomedicalAligner to generate soft tokens from cell
    centroids and variant embeddings, then generates predictions via
    BioMistral+LoRA with latent space conditioning.
    """
    predictions = []
    aligner = model["aligner"]
    llm = model["llm"]
    tokenizer = model["tokenizer"]
    device = model["device"]
    
    # Build lookup for variant embeddings
    var_lookup = _build_variant_lookup(mave_df)
    
    for ex in tqdm(examples, desc="scMultiPert MAVE"):
        # Extract gene and variant info from example
        gene = ex.gene_id
        variant_id = ex.embedding_key
        
        # Retrieve variant embedding
        var_emb = var_lookup.get(variant_id, None)
        if var_emb is None:
            continue
        
        # Use neutral cell prior (consistent with Module 5)
        neutral_cell = torch.zeros(256).to(device)  # Or compute from centroids
        
        # Generate soft tokens via aligner (Module 3 interface)
        with torch.no_grad():
            cell_batch = neutral_cell.unsqueeze(0)  # [1, 256]
            var_batch = var_emb.unsqueeze(0).to(device)  # [1, 320]
            soft_tokens, _ = aligner(cell_batch, var_batch)  # [1, 2, 4096]
        
        # Prepare input with soft token injection (Module 5 style)
        messages = ex.messages
        prompt = messages[1]["content"] if len(messages) > 1 else ""
        
        # Tokenize and inject soft tokens
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_embeds = llm.get_input_embeddings()(inputs.input_ids)
        
        # Find BIO_CONTEXT_TOKEN position and splice (Module 5 compatibility)
        bio_token_id = tokenizer.convert_tokens_to_ids(BIO_CONTEXT_TOKEN)
        positions = (inputs.input_ids == bio_token_id).nonzero(as_tuple=True)[1]
        
        if len(positions) > 0:
            pos = positions[0].item()
            # Replace with soft tokens
            prefix = input_embeds[:, :pos, :]
            suffix = input_embeds[:, pos+1:, :]
            # Trim suffix to maintain sequence length
            suffix = suffix[:, :suffix.size(1)-1, :]
            input_embeds = torch.cat([prefix, soft_tokens, suffix], dim=1)
        
        # Generate prediction
        with torch.no_grad():
            outputs = llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=100,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse predicted score from generated text
        pred_score = _parse_mave_score_from_text(generated_text)
        pred_class = _score_to_class(pred_score)
        
        # Get ground truth
        gt_row = mave_df[mave_df["variant_id"] == variant_id].iloc[0]
        
        predictions.append(MAVEPrediction(
            variant_id=variant_id,
            gene_symbol=gene,
            predicted_score=pred_score,
            true_score=gt_row["sge_score"],
            predicted_class=pred_class,
            true_class=gt_row["function_class"]
        ))
    
    return predictions


def per_protein_spearman(
    predictions: List[MAVEPrediction],
    mave_df: pd.DataFrame
) -> Dict[str, float]:
    """
    Compute Spearman correlation per protein (BRCA1, TP53, PTEN).
    
    ProteinGym-style metric: ρ between predicted and true SGE scores
    within each gene separately, then average.
    
    Args:
        predictions: List of MAVEPrediction objects.
        mave_df: Ground truth DataFrame for reference.
    
    Returns:
        Dict mapping gene_symbol -> Spearman ρ.
    """
    per_gene_scores = {}
    
    for gene in mave_df["gene_symbol"].unique():
        gene_preds = [p for p in predictions if p.gene_symbol == gene]
        
        if len(gene_preds) < 3:
            per_gene_scores[gene] = 0.0
            continue
        
        pred_scores = [p.predicted_score for p in gene_preds]
        true_scores = [p.true_score for p in gene_preds]
        
        rho, _ = spearmanr(pred_scores, true_scores)
        per_gene_scores[gene] = rho if not np.isnan(rho) else 0.0
    
    return per_gene_scores


def top_k_recall(
    predictions: List[MAVEPrediction],
    k: float = 0.10
) -> float:
    """
    Recall of top-K most harmful variants (LOF variants).
    
    Measures ability to identify the most deleterious mutations.
    
    Args:
        predictions: List of MAVEPrediction objects.
        k: Fraction of variants to consider (default 0.10 = top 10%).
    
    Returns:
        Recall@K: Fraction of true top-K variants identified in predicted top-K.
    """
    # Sort by true score (most negative = most harmful)
    sorted_by_true = sorted(predictions, key=lambda x: x.true_score)
    n_top = max(1, int(len(sorted_by_true) * k))
    true_top_k = set([p.variant_id for p in sorted_by_true[:n_top]])
    
    # Sort by predicted score
    sorted_by_pred = sorted(predictions, key=lambda x: x.predicted_score)
    pred_top_k = set([p.variant_id for p in sorted_by_pred[:n_top]])
    
    # Compute recall
    intersection = len(true_top_k.intersection(pred_top_k))
    recall = intersection / len(true_top_k) if len(true_top_k) > 0 else 0.0
    
    return recall


def directionality_check(predictions: List[MAVEPrediction]) -> float:
    """
    Binary accuracy: beneficial vs. non-beneficial (detrimental) classification.
    
    Simpler than 3-class: just predict if variant is functional (beneficial)
    vs. loss_of_function/intermediate (detrimental).
    
    Args:
        predictions: List of MAVEPrediction objects.
    
    Returns:
        Accuracy of binary directionality classification.
    """
    correct = 0
    total = 0
    
    for p in predictions:
        # Binary: functional = beneficial, LOF/intermediate = detrimental
        true_binary = 1 if p.true_class == "functional" else 0
        pred_binary = 1 if p.predicted_class == "functional" else 0
        
        if true_binary == pred_binary:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# SECTION 4 — scRNA Benchmark (GEARS-style)
# ---------------------------------------------------------------------------

def evaluate_scrna(
    model: Dict[str, any],
    test_examples: List[ChatMLExample],
    centroids: Dict[str, torch.Tensor],
    adata: ad.AnnData
) -> Dict[str, float]:
    """
    Execute GEARS-style evaluation on scRNA-seq response prediction.
    
    Evaluates transcriptional response generation by comparing top-K
    differentially expressed genes in generated text vs. actual centroids.
    
    Args:
        model: scMultiPert model dictionary.
        test_examples: Filtered ChatMLExample list with modality=="scrna".
        centroids: Module 4 perturbation centroids.
        adata: Module 1 AnnData with raw counts for DE gene calculation.
    
    Returns:
        Dictionary of metrics:
            - "gene_set_f1": Float (F1 between generated and true top-K genes)
            - "per_perturbation_f1": Dict[str, float] (per-condition F1)
            - "id_gene_set_f1": Float (F1 on in-distribution genes)
            - "ood_gene_set_f1": Float (F1 on out-of-distribution genes)
    """
    # Filter to scRNA examples
    scrna_examples = [ex for ex in test_examples if ex.modality == "scrna"]
    
    if len(scrna_examples) == 0:
        raise EvaluationError("No scRNA examples found in test corpus")
    
    per_perturbation_f1 = {}
    id_f1_scores = []
    ood_f1_scores = []
    
    for ex in tqdm(scrna_examples, desc="scRNA evaluation"):
        perturbation = ex.embedding_key
        
        # Generate response using scMultiPert
        generated_text = _generate_scrna_response(model, ex, centroids)
        
        # Extract mentioned genes from generated text
        generated_genes = _extract_gene_mentions(generated_text, adata.var_names)
        
        # Get true top-K genes from actual centroid/AnnData
        true_genes = _get_top_k_genes_from_adata(
            adata, perturbation, k=5
        )
        
        # Compute F1
        f1 = _compute_gene_set_f1(generated_genes, true_genes, k=5)
        per_perturbation_f1[perturbation] = f1
        
        # Classify as ID or OOD based on gene vocabulary
        # (Would need train gene set from m4_data, simplified here)
        # Append to appropriate list for ID vs OOD gap calculation
    
    results = {
        "gene_set_f1": np.mean(list(per_perturbation_f1.values())),
        "per_perturbation_f1": per_perturbation_f1,
        "id_gene_set_f1": np.mean(id_f1_scores) if id_f1_scores else 0.0,
        "ood_gene_set_f1": np.mean(ood_f1_scores) if ood_f1_scores else 0.0
    }
    
    return results


def gene_set_f1(
    predicted_genes: List[str],
    true_genes: List[str],
    k: int = 5
) -> float:
    """
    Compute F1 score between top-K predicted and true gene sets.
    
    GEARS-style metric: Measures overlap between generated gene mentions
    and actual differentially expressed genes.
    
    Args:
        predicted_genes: List of gene symbols mentioned in generated text.
        true_genes: List of actual top-K DE genes.
        k: Number of top genes to consider.
    
    Returns:
        F1 score (harmonic mean of precision and recall).
    """
    pred_set = set(predicted_genes[:k])
    true_set = set(true_genes[:k])
    
    if len(pred_set) == 0 and len(true_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(true_set) == 0:
        return 0.0
    
    intersection = len(pred_set.intersection(true_set))
    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def id_vs_ood_gap(
    id_f1_scores: List[float],
    ood_f1_scores: List[float]
) -> float:
    """
    Compute generalization gap between in-distribution and OOD genes.
    
    Args:
        id_f1_scores: F1 scores on training-set genes.
        ood_f1_scores: F1 scores on held-out (test-set) genes.
    
    Returns:
        Gap value (ID_F1 - OOD_F1). Lower is better (good generalization).
    """
    id_mean = np.mean(id_f1_scores) if id_f1_scores else 0.0
    ood_mean = np.mean(ood_f1_scores) if ood_f1_scores else 0.0
    
    return id_mean - ood_mean


# ---------------------------------------------------------------------------
# SECTION 5 — Generalist LLM Benchmark
# ---------------------------------------------------------------------------

def evaluate_medqa(
    scmp_model: Dict[str, any],
    baseline_model: Dict[str, any],
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate medical QA capability retention (MedQA-USMLE subset).
    
    No-regression check: Ensures fine-tuning didn't harm generalist
    medical knowledge compared to base BioMistral.
    
    Args:
        scmp_model: scMultiPert model dictionary.
        baseline_model: Base BioMistral zero-shot model.
        n_samples: Number of MedQA questions to evaluate.
    
    Returns:
        Dictionary with "scmp" and "baseline" accuracy scores.
    """
    # Load MedQA subset (would need actual dataset loading)
    medqa_questions = _load_medqa_subset(n_samples)
    
    scmp_correct = 0
    baseline_correct = 0
    
    for q in tqdm(medqa_questions, desc="MedQA evaluation"):
        # Evaluate scMultiPert
        scmp_answer = _generate_answer(scmp_model, q["question"], q["choices"])
        if scmp_answer == q["correct"]:
            scmp_correct += 1
        
        # Evaluate baseline
        baseline_answer = _generate_answer(baseline_model, q["question"], q["choices"])
        if baseline_answer == q["correct"]:
            baseline_correct += 1
    
    return {
        "scmp": scmp_correct / len(medqa_questions),
        "baseline": baseline_correct / len(medqa_questions)
    }


def evaluate_pubmedqa(
    scmp_model: Dict[str, any],
    baseline_model: Dict[str, any],
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate biomedical literature understanding (PubMedQA).
    
    Tests ability to answer yes/no/maybe questions from PubMed abstracts.
    
    Args:
        scmp_model: scMultiPert model dictionary.
        baseline_model: Base BioMistral zero-shot model.
        n_samples: Number of PubMedQA questions to evaluate.
    
    Returns:
        Dictionary with "scmp" and "baseline" accuracy scores.
    """
    # Load PubMedQA subset
    pubmedqa_questions = _load_pubmedqa_subset(n_samples)
    
    scmp_correct = 0
    baseline_correct = 0
    
    for q in tqdm(pubmedqa_questions, desc="PubMedQA evaluation"):
        scmp_answer = _generate_answer(scmp_model, q["question"], [])
        if scmp_answer == q["correct"]:
            scmp_correct += 1
        
        baseline_answer = _generate_answer(baseline_model, q["question"], [])
        if baseline_answer == q["correct"]:
            baseline_correct += 1
    
    return {
        "scmp": scmp_correct / len(pubmedqa_questions),
        "baseline": baseline_correct / len(pubmedqa_questions)
    }


def generalist_mave_spearman(
    generalist_model: Dict[str, any],
    test_examples: List[ChatMLExample],
    mave_df: pd.DataFrame
) -> float:
    """
    Evaluate generalist LLM (Llama-3/GPT-4o) on same MAVE test set.
    
    Closes the gap analysis: Compares specialized scMultiPert against
    general-purpose models on the biological task.
    
    Args:
        generalist_model: Model dictionary from load_generalist_llm().
        test_examples: MAVE test examples.
        mave_df: Ground truth DataFrame.
    
    Returns:
        Mean Spearman correlation across proteins.
    """
    # Reuse evaluate_mave logic
    results = evaluate_mave(
        model=generalist_model,
        test_examples=test_examples,
        mave_df=mave_df,
        var_embeddings=None,
        var_ids=[],
        centroids=None
    )
    
    return results.get("mean_spearman", 0.0)


# ---------------------------------------------------------------------------
# SECTION 6 — Interpretability Analysis
# ---------------------------------------------------------------------------

def extract_attention_heatmaps(
    aligner: BiomedicalAligner,
    test_batch: List[Tuple[torch.Tensor, torch.Tensor, str, str]]
) -> Dict[str, np.ndarray]:
    """
    Extract cross-attention weights from BiomedicalAligner for visualization.
    
    Args:
        aligner: Module 3 BiomedicalAligner instance.
        test_batch: List of (cell_emb, var_emb, cell_label, var_id) tuples.
    
    Returns:
        Dictionary mapping "cell_label|var_id" -> attention weight matrix.
    """
    aligner.eval()
    heatmaps = {}
    
    for cell_emb, var_emb, cell_label, var_id in test_batch:
        with torch.no_grad():
            # Ensure batch dimension
            if cell_emb.ndim == 1:
                cell_emb = cell_emb.unsqueeze(0)
            if var_emb.ndim == 1:
                var_emb = var_emb.unsqueeze(0)
            
            # Forward through aligner
            soft_tokens, attn_weights = aligner(cell_emb, var_emb)
            
            # attn_weights shape: [B, 1, 1] for single-head attention
            # Store for visualization
            key = f"{cell_label}|{var_id}"
            heatmaps[key] = attn_weights.cpu().numpy()
    
    return heatmaps


def lof_softtoken_classification(
    aligner: BiomedicalAligner,
    centroids: Dict[str, torch.Tensor],
    var_embeddings: torch.Tensor,
    mave_df: pd.DataFrame,
    n_samples: int = 500
) -> float:
    """
    Classify LOF vs. functional using ONLY soft tokens, no text generation.
    
    Tests if the aligned latent space inherently captures functional
    variant information without decoder assistance.
    
    Args:
        aligner: Module 3 BiomedicalAligner.
        centroids: Perturbation centroids (use neutral for pure variant test).
        var_embeddings: Module 2 variant embeddings.
        mave_df: Ground truth with function_class labels.
        n_samples: Number of variants to test.
    
    Returns:
        F1 score of binary LOF vs. functional classification from soft tokens.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Sample variants with balanced classes
    lof_variants = mave_df[mave_df["function_class"] == "loss_of_function"]
    func_variants = mave_df[mave_df["function_class"] == "functional"]
    
    n_per_class = min(n_samples // 2, len(lof_variants), len(func_variants))
    
    sampled_lof = lof_variants.sample(n=n_per_class, random_state=42)
    sampled_func = func_variants.sample(n=n_per_class, random_state=42)
    sampled = pd.concat([sampled_lof, sampled_func])
    
    # Build soft tokens for each variant (with neutral cell context)
    device = aligner.device
    neutral_cell = torch.zeros(256).to(device)
    
    soft_token_list = []
    labels = []
    
    for _, row in sampled.iterrows():
        # Find variant embedding
        var_idx = _find_variant_index(row["variant_id"], mave_df)
        if var_idx is None:
            continue
        
        var_emb = var_embeddings[var_idx].to(device)
        
        with torch.no_grad():
            cell_batch = neutral_cell.unsqueeze(0)
            var_batch = var_emb.unsqueeze(0)
            soft_tokens, _ = aligner(cell_batch, var_batch)
        
        # Use attended cell token (index 0) as representation
        representation = soft_tokens[0, 0, :].cpu().numpy()
        soft_token_list.append(representation)
        
        # Binary label: 1 for LOF, 0 for functional
        labels.append(1 if row["function_class"] == "loss_of_function" else 0)
    
    # Train simple classifier on soft tokens
    X = np.array(soft_token_list)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    f1 = f1_score(y_test, y_pred)
    return f1


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------

def _validate_benchmark_inputs(
    checkpoints: Dict[str, str],
    m2_outputs: Dict[str, any],
    m4_data: Dict[str, any]
) -> None:
    """Validate that all required inputs are present and correctly formatted."""
    required_ckpts = ["scmultipert", "aligner"]
    for key in required_ckpts:
        if key not in checkpoints:
            raise CheckpointError(f"Missing checkpoint key: {key}")
    
    required_m2 = ["var_embeddings", "var_ids", "cell_embeddings", "cell_metadata"]
    for key in required_m2:
        if key not in m2_outputs:
            raise EvaluationError(f"Missing m2_outputs key: {key}")
    
    required_m4 = ["test_path", "centroids", "gene_vocabulary"]
    for key in required_m4:
        if key not in m4_data:
            raise EvaluationError(f"Missing m4_data key: {key}")


def _load_test_corpus(test_path: str) -> List[ChatMLExample]:
    """Load test examples from Module 4 JSONL output."""
    examples = []
    with open(test_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(ChatMLExample(**data))
    return examples


def _tag_results(
    results: Dict[str, float],
    model_name: str,
    task: str
) -> List[BenchmarkResult]:
    """Convert metric dictionary to list of BenchmarkResult objects."""
    tagged = []
    for metric_name, metric_value in results.items():
        tagged.append(BenchmarkResult(
            model_name=model_name,
            task=task,
            metric_name=metric_name,
            metric_value=metric_value if isinstance(metric_value, float) else 0.0,
            metadata={"raw_value": metric_value} if not isinstance(metric_value, float) else None
        ))
    return tagged


def _results_to_dataframe(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Convert BenchmarkResult list to DataFrame."""
    records = []
    for r in results:
        record = {
            "model_name": r.model_name,
            "task": r.task,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
        }
        if r.metadata:
            record["metadata"] = str(r.metadata)
        records.append(record)
    return pd.DataFrame(records)


def _save_benchmark_results(df: pd.DataFrame, output_dir: str) -> None:
    """Save results to CSV and JSON."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
    
    # Also save as JSON for programmatic access
    df.to_json(f"{output_dir}/benchmark_results.json", orient="records", indent=2)


def _build_variant_lookup(mave_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """Build lookup dictionary for variant embeddings (simplified)."""
    # In practice, this would map variant_id to its embedding tensor
    # For now, return empty - actual implementation would use var_embeddings
    return {}


def _parse_mave_score_from_text(text: str) -> float:
    """Extract predicted SGE score from generated text."""
    # Simple regex-based extraction
    import re
    match = re.search(r'sge score[:\s]+(-?\d+\.?\d*)', text.lower())
    if match:
        return float(match.group(1))
    return 0.0


def _score_to_class(score: float) -> str:
    """Convert score to functional class (Module 4 compatible thresholds)."""
    if score < -1.0:
        return "loss_of_function"
    elif score < -0.5:
        return "intermediate"
    else:
        return "functional"


def _extract_mave_labels(mave_df: pd.DataFrame, var_ids: List[str]) -> pd.Series:
    """Extract SGE scores aligned with var_ids for MLP training."""
    score_map = dict(zip(mave_df["variant_id"], mave_df["sge_score"]))
    scores = [score_map.get(vid, 0.0) for vid in var_ids]
    return pd.Series(scores)


def _sample_test_batch(
    examples: List[ChatMLExample],
    n: int = 100
) -> List[Tuple[torch.Tensor, torch.Tensor, str, str]]:
    """Sample test batch for attention extraction."""
    # Simplified - would actually pair cell and variant embeddings
    return []


def _evaluate_mave_bm25(
    model: Dict[str, any],
    examples: List[ChatMLExample],
    mave_df: pd.DataFrame
) -> List[MAVEPrediction]:
    """BM25-based MAVE evaluation (lexical retrieval)."""
    # Implementation would use BM25 scores as predictions
    return []


def _evaluate_mave_mlp(
    model: Dict[str, any],
    examples: List[ChatMLExample],
    mave_df: pd.DataFrame,
    var_embeddings: torch.Tensor,
    var_ids: List[str]
) -> List[MAVEPrediction]:
    """MLP-based MAVE evaluation (numeric regression)."""
    # Implementation would use MLP predictions
    return []


def _evaluate_mave_llm(
    model: Dict[str, any],
    examples: List[ChatMLExample],
    mave_df: pd.DataFrame
) -> List[MAVEPrediction]:
    """Generic LLM MAVE evaluation (text-only)."""
    # Implementation for zero-shot LLMs
    return []


def _generate_scrna_response(
    model: Dict[str, any],
    example: ChatMLExample,
    centroids: Dict[str, torch.Tensor]
) -> str:
    """Generate scRNA response using scMultiPert."""
    # Implementation would use model generation with soft tokens
    return ""


def _extract_gene_mentions(text: str, var_names: pd.Index) -> List[str]:
    """Extract gene symbols mentioned in generated text."""
    # Simple extraction based on vocabulary matching
    return []


def _get_top_k_genes_from_adata(
    adata: ad.AnnData,
    perturbation: str,
    k: int = 5
) -> List[str]:
    """Get top K DE genes for a perturbation from AnnData."""
    # Implementation would compute from adata
    return []


def _compute_gene_set_f1(
    predicted: List[str],
    true: List[str],
    k: int = 5
) -> float:
    """Compute F1 between gene sets."""
    return gene_set_f1(predicted, true, k)


def _load_medqa_subset(n: int) -> List[Dict]:
    """Load MedQA subset."""
    return []


def _load_pubmedqa_subset(n: int) -> List[Dict]:
    """Load PubMedQA subset."""
    return []


def _generate_answer(
    model: Dict[str, any],
    question: str,
    choices: List[str]
) -> str:
    """Generate answer for QA evaluation."""
    return ""


def _find_variant_index(variant_id: str, mave_df: pd.DataFrame) -> Optional[int]:
    """Find index of variant in embeddings."""
    return None


# ---------------------------------------------------------------------------
# MODULE ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 6: Import and use run_benchmarking() with Module 2-5 outputs.")
    print("See docstring for integration details.")