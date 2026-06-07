# Single-cell Immune & TCR Repertoire Analysis

## Overview

This repository contains computational workflows for single-cell RNA-seq and TCR repertoire analysis in paediatric immune datasets.

The project investigates:
- Immune ageing dynamics
- T-cell compositional changes
- Differential abundance patterns
- TCR clonotype expansion
- Repertoire diversity and transcriptomic integration

Analyses were performed in Linux-based HPC environments using reproducible Python and R workflows.

---

## Analysis Components

### Single-cell RNA-seq Analysis
- Quality control and preprocessing
- Batch correction and integration using scVI
- Dimensionality reduction and clustering
- UMAP visualisation
- Cell type annotation using canonical marker expression

### Differential Abundance Analysis
- Milo neighbourhood-based differential abundance testing
- scCODA compositional modelling
- Age- and sex-associated abundance analysis

### TCR Repertoire Analysis
- TCR integration using Dandelion
- Clonotype expansion analysis
- Clone size categorisation
- Repertoire diversity analysis
- TCR-transcriptome integration

### Statistical Visualisation
- UMAP visualisation
- Differential abundance plots
- Clonotype distribution analysis
- Composition and repertoire visualisation

---

## Tools & Packages

### Python
- Scanpy
- scVI-tools
- Dandelion
- Milo
- PertPy
- pandas
- NumPy
- matplotlib
- seaborn

### R
- MiloR
- tidyverse
- ggplot2

### Infrastructure
- Linux
- HPC environments (Bunya)
- bash
- conda
- Jupyter Notebook

---

## Repository Structure

```text
notebooks/      Analysis notebooks
scripts/        Reusable workflow scripts
Results/        Analysis outputs and figures
dandelion/      TCR repertoire workflows
workflow/       Pipeline diagrams and workflow notes
environment/    Package and environment specifications
```

---

## Workflow Overview

```text
Raw scRNA-seq + VDJ data
                ↓
Quality control and filtering
                ↓
Batch correction / integration (scVI)
                ↓
Dimensionality reduction and clustering
                ↓
Cell type annotation
                ↓
Differential abundance analysis
        ↙                     ↘
     Milo                  scCODA
                ↓
TCR integration (Dandelion)
                ↓
Clonotype expansion analysis
                ↓
Clone size categorisation
                ↓
Repertoire diversity analysis
                ↓
Statistical visualisation and interpretation
```

---

## Skills Demonstrated

- Single-cell RNA-seq analysis
- Cell type annotation
- Differential abundance modelling
- TCR clonotype analysis
- Immune repertoire analysis
- Statistical visualisation
- HPC workflow management
- Reproducible computational biology pipelines

---

## Notes

Raw sequencing datasets are not included in this repository.

This repository focuses on reproducible downstream computational workflows and analyses.
