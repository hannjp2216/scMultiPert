# scMultiPert

A perturbation-aware LLM framework for multimodal in-silico perturbation 
modelling, developed as part of GSoC 2026 with EMBL-EBI.

## Overview

scMultiPert fine-tunes BioMistral-7B to act as a knowledge-integration 
layer across three perturbation data modalities: CRISPR screens (DepMap), 
MAVE variant-effect data (BRCA1, TP53, PTEN), and scRNA-seq perturbation 
responses (Norman et al. 2019). It uses a specialized BiomedicalAligner 
to project biological embeddings from Geneformer and ESM-2 into the LLM's 
latent space via soft token injection.

> **Status:** Pseudocode modules uploaded. Full implementation begins 
> GSoC 2026 Community Bonding Period (May 4).

## Repository Structure
```
scMultiPert/
├── module1_preprocessing.py   # Data ingestion: scRNA-seq, CRISPR, MAVE
├── module2_encoders.py        # Geneformer (256-d) + ESM-2 8M (320-d) encoding
├── module3_alignment.py       # BiomedicalAligner + soft token projection
├── module4_corpus.py          # ChatML corpus builder with gene-based splits
├── module5_finetuning.py      # QLoRA two-stage fine-tuning + SoftTokenCollator
├── module6_benchmarking.py    # ProteinGym + GEARS-style evaluation suite
├── validate_pipeline.py       # Read-only integrity checks between all stages
└── README.md
```

## Pipeline Architecture
```
Module 1 (Preprocessing)
    └── Module 2 (Encoders: Geneformer + ESM-2)
            └── Module 3 (BiomedicalAligner)
            └── Module 4 (Corpus Builder)
                    └── Module 5 (QLoRA Fine-tuning)
                            └── Module 6 (Benchmarking)

validate_pipeline.py runs between every stage.
```

## Key Design Decisions

- **Gene-based train/test splitting** — no gene appears in both splits, 
  including component genes of combinatorial perturbations (e.g. CDKN1A+FOXA1)
- **Dual soft token output** — [Attended_Cell, Raw_Variant] preserves 
  variant-specific signal that cross-attention might attenuate
- **Neutral priors computed from training keys only** — prevents test-set 
  leakage through embedding averages during Stage 2 training
- **CPU→GPU handoff** — Module 2 stores all embeddings on CPU; 
  SoftTokenCollator moves only batch-sized tensors to GPU at collation time

## Dependencies
```bash
# Core ML
torch>=2.1.0
transformers>=4.35.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0

# Biological data
scanpy>=1.9.0
anndata>=0.9.0
geneformer @ git+https://huggingface.co/ctheodoris/Geneformer

# Evaluation
scipy>=1.11.0
scikit-learn>=1.3.0
rank-bm25>=0.2.2

# Experiment tracking
wandb>=0.15.0
```

## Hardware Requirements

| Stage | GPU Memory | Estimated Time |
|---|---|---|
| Modules 1–2 (Encoding) | 8–16 GB | 2–4 hours |
| Module 3 (Alignment training) | 24 GB | 4–6 hours |
| Module 5 Stage 1 (Text LoRA) | 16 GB | 6–8 hours |
| Module 5 Stage 2 (Joint tuning) | 40 GB | 12–16 hours |
| Module 6 (Benchmarking) | 24 GB | 3–4 hours |

Development: single A100 (40GB). Production: 2×A6000 (48GB each).

## GSoC 2026

This project is being developed under Google Summer of Code 2026 
with the EMBL-EBI organization.

- **Contributor:** Hannia Isabel Juárez Pérez
- **Mentors:** Alexey Sokolov, Kirill Tsukanov, Aleksandr Zakirov
- **Organization:** EMBL-EBI

## License

Apache 2.0

