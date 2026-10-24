
# TMWF — Transformer-Based Multi-Tab Website Fingerprinting | Research Reproduction

**A reproduction of the TMWF research paper (ACM CCS 2023)**

This repository contains an experimental reproduction of **Transformer-based Model for Multi-tab Website Fingerprinting Attack (TMWF)**, originally published at ACM CCS 2023.

The objective is to reproduce selected experiments from the original paper, evaluate the model on Tor traffic datasets, and compare the obtained results with the published findings.

### Reproduction Highlights

- Implemented and evaluated the original **DFNet–Transformer architecture**.
- Reproduced **Appendix F (Table 7)** and **Appendix H (Table 9)**.
- Completed two independent training runs of **50 epochs each**.
- Trained on **20,000 and 40,000 synthetic multi-tab traces**, respectively.
- Evaluated synthetic and real-world multi-tab Tor traffic.
- Verified all **36 primary published-versus-reproduced metrics**.
- Obtained results within **4 percentage points for 35 of 36 metrics**.
- Preserved evaluation logs, configuration details, and reproducibility evidence.

> **Reproduction scope:** This repository reproduces selected experiments from Appendices F and H, not every experiment in the original paper. Numerical agreement does not imply exact or bitwise reproduction.

---

## Research Paper and Documentation

**Original Paper:** Transformer-based Model for Multi-tab Website Fingerprinting Attack (ACM CCS 2023)

**Original Implementation:** [jzx-bupt/TMWF](https://github.com/jzx-bupt/TMWF)

**Upstream Git Commit:** `f593e8afba22750edae23c0c0c32ddf6a62772d4`

### Full Reproduction Report

**[View the Complete Reproduction Report (PDF)](docs/TMWF_REPRODUCTION_REPORT.pdf)**

The 23-page report documents:

- Research background and threat model
- Original TMWF architecture
- Dataset preparation and synthetic trace generation
- Training and evaluation methodology
- Published-versus-reproduced results
- Implementation choices and deviations
- Experimental limitations and reproducibility evidence

[Read the Markdown Report](docs/TMWF_REPRODUCTION_REPORT.md)

---

## 1. Research Overview

Website Fingerprinting (WF) is a traffic-analysis technique that attempts to identify websites from observable traffic patterns even when communication is encrypted.

In the Tor network, an observer monitoring the connection between a client and its entry relay may analyze traffic direction and timing patterns without decrypting the underlying content.

Traditional website fingerprinting generally focuses on **single-tab browsing**.

Multi-tab browsing introduces additional challenges because traffic from multiple simultaneously active webpages becomes interleaved, making individual website identification more difficult.

TMWF addresses this problem using a deep-learning architecture combining:

1. **DFNet** for traffic feature extraction.
2. **Transformer Encoder** for contextual representation.
3. **Transformer Decoder** for multi-page prediction.

The model predicts website classes across multiple output slots.

---

## 2. TMWF Model Architecture

![TMWF Architecture](docs/figures/tmwf_architecture.png)

*Figure 1. Architecture of the reproduced TMWF model.*

### Model Configuration

| Component | Configuration |
|---|---|
| Input representation | Tor-cell direction sequence |
| Input sequence length | 30,720 |
| Feature extractor | DFNet |
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer encoder layers | 2 |
| Transformer decoder layers | 2 |
| Feedforward dimension | 1,024 |
| Output queries | 6 |
| Classes per query | 51 |
| Dropout | 0.1 |

The input is represented using traffic directions, with positive and negative values indicating opposite traffic directions. Sequences are truncated or zero-padded to the required length.

The model produces **six classification outputs**, with each output predicting one of 51 classes.

The additional class represents the unmonitored/no-tab category according to the original implementation.

Training uses ordered six-slot supervision and summed cross-entropy loss.

---

## 3. Experimental Pipeline

![Experimental Pipeline](docs/figures/tmwf_experimental_pipeline.png)

*Figure 2. Data preparation, training, and evaluation pipelines for Appendices F and H.*

The reproduction uses the original Tor Browser (TBB) datasets provided with the released implementation.

### Original Single-Tab Dataset

| Partition | Samples |
|---|---:|
| Training | 8,000 |
| Testing | 2,000 |

The original training partition contains 4,000 monitored and 4,000 unmonitored traces.

Synthetic training datasets are generated from the original training partition. Synthetic evaluation datasets are generated from the held-out testing partition.

The two reproduced experimental configurations use separate training runs.

### Appendix F

- 20,000 synthetic six-tab training traces
- 5,000 evaluation traces each for 2-, 4-, and 6-tab settings
- Duration-overlap synthesis
- 50 training epochs

### Appendix H

- 40,000 synthetic six-tab training traces
- Ratio-synthetic evaluation: 1,000 each for 2, 3, and 4 tabs
- Delay-synthetic evaluation: 1,000 each for 2, 3, and 4 tabs
- Real-world evaluation: 1,000 each for 2, 3, and 4 tabs
- 50 training epochs

Real-world and delay-synthetic evaluation data were not used to train the Appendix H model.

---

## 4. Experimental Results

All results below are from the final **epoch-50 checkpoints**.

The published values are compared against the independently reproduced values using the original evaluation metrics.

### Appendix F — Table 7

**Overall Basic Metrics**

| Metric | Tabs | Published | Reproduced |
|---|---:|---:|---:|
| Accuracy | 2 | 55.6% | **55.9%** |
| Precision | 2 | 71.9% | **71.9%** |
| Recall | 2 | 75.2% | **72.0%** |
| Accuracy | 4 | 64.7% | **64.3%** |
| Precision | 4 | 74.2% | **73.7%** |
| Recall | 4 | 73.4% | **71.9%** |
| Accuracy | 6 | 65.1% | **63.0%** |
| Precision | 6 | 77.8% | **77.4%** |
| Recall | 6 | 67.6% | **64.6%** |

**Overall Advanced Metrics**

| Metric | Tabs | Published | Reproduced |
|---|---:|---:|---:|
| Accuracy | 2 | 19.3% | **21.3%** |
| Precision | 2 | 28.3% | **30.9%** |
| Recall | 2 | 40.2% | **40.3%** |
| Accuracy | 4 | 17.7% | **18.4%** |
| Precision | 4 | 25.4% | **25.9%** |
| Recall | 4 | 29.1% | **30.1%** |
| Accuracy | 6 | 52.1% | **50.0%** |
| Precision | 6 | 64.8% | **63.6%** |
| Recall | 6 | 59.9% | **57.1%** |

All 18 Appendix F primary metrics are within **3.2 percentage points** of the published results.

![Appendix F Results](docs/figures/tmwf_table7_comparison.png)

*Figure 3. Comparison of published and reproduced Appendix F / Table 7 results.*

### Appendix H — Table 9

**Overall Basic Metrics**

| Evaluation Dataset | Metric | Published | Reproduced |
|---|---|---:|---:|
| Mixed Ratio-Synthetic | Accuracy | 61.3% | **60.1%** |
| Mixed Ratio-Synthetic | Precision | 75.3% | **75.9%** |
| Mixed Ratio-Synthetic | Recall | 73.5% | **71.2%** |
| Mixed Delay-Synthetic | Accuracy | 31.6% | **25.4%** |
| Mixed Delay-Synthetic | Precision | 63.8% | **60.9%** |
| Mixed Delay-Synthetic | Recall | 31.3% | **28.6%** |
| Mixed Real-World | Accuracy | 23.0% | **21.0%** |
| Mixed Real-World | Precision | 58.9% | **56.4%** |
| Mixed Real-World | Recall | 24.8% | **22.3%** |
| Real 2-Tab | Accuracy | 12.7% | **9.7%** |
| Real 2-Tab | Precision | 44.3% | **41.5%** |
| Real 2-Tab | Recall | 15.1% | **11.1%** |
| Real 3-Tab | Accuracy | 16.0% | **14.8%** |
| Real 3-Tab | Precision | 54.3% | **52.4%** |
| Real 3-Tab | Recall | 17.3% | **15.8%** |
| Real 4-Tab | Accuracy | 34.4% | **31.8%** |
| Real 4-Tab | Precision | 63.2% | **61.0%** |
| Real 4-Tab | Recall | 35.6% | **32.6%** |

![Appendix H Results](docs/figures/tmwf_table9_comparison.png)

*Figure 4. Comparison of published and reproduced Appendix H / Table 9 results.*

### Main Observations

- The ratio-synthetic results show close agreement with the original paper.
- Real-world performance is substantially lower than ratio-synthetic performance, consistent with the generalization gap reported in the paper.
- The largest primary-metric difference occurs in mixed delay-synthetic Basic Accuracy: **25.4% reproduced versus 31.6% published**, a difference of **−6.2 percentage points**.
- Overall, **35 of the 36 primary metrics** are within 4 percentage points of the published results.

These comparisons are descriptive single-run findings, not evidence of statistical equivalence. The metrics are not statistically independent.

---

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch 2.8.0 |
| Python | 3.9.6 |
| NumPy | 1.26.4 |
| Hardware backend | Apple Metal Performance Shaders (MPS) |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Batch size | 80 |
| Epochs | 50 |
| Loss | Summed six-slot CrossEntropyLoss |
| Random seed | 2023 |
| Checkpoint selection | Final epoch (50) |

Both model runs were trained from scratch.

The final epoch was used for evaluation, rather than selecting the epoch with the best validation performance.

---

## 6. Project Structure

```text
TMWF-Reproduction/
├── code/
│   ├── models.py
│   ├── metrics.py
│   ├── appendix_f_data.py
│   ├── appendix_h_data.py
│   ├── run_appendix_f_synthesis.py
│   ├── run_appendix_h_synthesis.py
│   ├── run_appendix_h_delay_synthesis.py
│   ├── train_appendix_f.py
│   ├── train_appendix_h.py
│   ├── evaluate_appendix_f.py
│   ├── evaluate_appendix_h.py
│   ├── evaluate_appendix_h_delay.py
│   └── ...
├── dataset/
│   └── Original tracked dataset archives
├── docs/
│   ├── TMWF_REPRODUCTION_REPORT.pdf
│   ├── TMWF_REPRODUCTION_REPORT.md
│   └── figures/
│       ├── tmwf_architecture.png
│       ├── tmwf_experimental_pipeline.png
│       ├── tmwf_table7_comparison.png
│       └── tmwf_table9_comparison.png
├── outputs/
│   └── reproducibility/
│       └── Archived evaluation logs and manifests
└── README.md
```

Generated traffic datasets, training checkpoints, and local Python environments are excluded from regular Git tracking due to their size.

---

## 7. Reproducibility and Evaluation

The experimental reproduction was performed on macOS using Python 3.9.6 and PyTorch 2.8.0 with MPS acceleration.

The following commands describe evaluation in the **original prepared local environment**.

They require:

- The original TMWF datasets to be extracted.
- The generated Appendix F/H datasets to be present.
- The trained epoch-50 checkpoints to be available.
- A compatible Python environment with the required dependencies.

### Appendix F Evaluation

```bash
.venv/bin/python -B code/evaluate_appendix_f.py
```

### Appendix H — Ratio and Real-World Evaluation

```bash
.venv/bin/python -B code/evaluate_appendix_h.py --evaluate
```

### Appendix H — Delay-Synthetic Evaluation

```bash
.venv/bin/python -B code/evaluate_appendix_h_delay.py --evaluate
```

### Results Verification

The complete evaluation outputs and integrity records are stored under:

`outputs/reproducibility/`

The archived evidence contains:

- Three successful evaluator logs
- Configuration and command records
- Checkpoint and file fingerprints
- Integrity-check results
- Published and reproduced metric comparisons

**Important:** This GitHub repository does not currently contain the complete generated datasets or trained checkpoints. The commands above therefore are not directly runnable from a fresh clone without reconstructing or obtaining those artifacts.

A fully verified clean-machine installation procedure and dependency lockfile have not yet been established.

---

## 8. Implementation Notes and Limitations

### Delay-Synthetic Reproduction

The Appendix H delay experiment uses a **copy-isolated implementation of the released delay-merging algorithm**.

The released implementation can mutate timestamp arrays in place, causing contamination when source traces are reused.

This reproduction independently copies each selected source occurrence to prevent such contamination while preserving the released delay scheduling and random-draw behavior.

Consequently, this is **not literal execution of the released generator**.

The original minor timestamp reversals were preserved. Delay annotations do not independently establish traffic-label correspondence.

The mixed-delay accuracy discrepancy remains unresolved; it should not be attributed to copy isolation without a controlled comparison.

### Other Limitations

- Experiments were conducted using a single recorded training run per configuration.
- Seeds do not guarantee bitwise-identical results across hardware backends.
- Some original dataset provenance properties were not independently verifiable.
- The reproduced results are numerically close but not identical to the published findings.
- The reproduction covers selected Appendix F/H experiments, not the entire original paper.

For the full methodology and discussion, see the [reproduction report](docs/TMWF_REPRODUCTION_REPORT.pdf).

---

## 9. Student Reproduction Contributions

This project builds upon the original research and open-source implementation.

The reproduction-specific work includes:

- Adapting the original implementation to a macOS/PyTorch MPS environment.
- Preparing synthetic-data generation workflows.
- Developing Appendix F/H dataset loaders and evaluation adapters.
- Implementing structural and integrity audits.
- Investigating timestamp mutation in the released delay-merging algorithm.
- Implementing and documenting copy-isolated delay synthesis.
- Training and evaluating the reproduced models.
- Archiving evaluation logs, checkpoints' provenance metadata, and numerical comparisons.
- Preparing the final reproducibility report and research figures.

The original TMWF architecture, core methodology, and research contributions belong to the original authors.

---

## 10. References and Attribution

**Original Research:**  
*Transformer-based Model for Multi-tab Website Fingerprinting Attack.* ACM Conference on Computer and Communications Security (CCS), 2023.

**Original Source Repository:**  
https://github.com/jzx-bupt/TMWF

**Upstream Commit:**  
`f593e8afba22750edae23c0c0c32ddf6a62772d4`

This repository preserves the original implementation's history and adds reproduction-specific scripts, experimental analysis, and documentation.

Please cite the original paper when using or discussing the TMWF method.

---

## Reproduction Summary

| Item | Result |
|---|---|
| Appendix F / Table 7 | Reproduced |
| Appendix H / Table 9 | Reproduced with documented deviations |
| Primary metrics verified | 36 / 36 |
| Metrics within 4 percentage points | 35 / 36 |
| Archived evaluation runs | 3 successful |
| Final research report | 23-page PDF |

**[Read the Full TMWF Reproduction Report](docs/TMWF_REPRODUCTION_REPORT.pdf)**
