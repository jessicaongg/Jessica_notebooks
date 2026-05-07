# Methodology

This repository contains single-cell transcriptomic and immune repertoire analysis workflows for paediatric immune datasets.

## Workflow Summary

### 1. Single-cell preprocessing
Single-cell RNA-seq datasets were processed using Scanpy. Low-quality cells and genes were filtered based on quality control metrics.

### 2. Batch correction and integration
scVI was used for latent embedding and integration across multiple samples and batches.

### 3. Dimensionality reduction and clustering
Neighbour graph construction, UMAP dimensionality reduction and Leiden clustering were performed for cell-state identification.

### 4. Cell type annotation
Cell populations were annotated using canonical immune marker expression profiles.

### 5. Differential abundance analysis
Differential abundance analyses were performed using:
- Milo neighbourhood-based testing
- scCODA compositional modelling

Age and sex covariates were incorporated where appropriate.

### 6. TCR repertoire analysis
TCR sequencing data were integrated using Dandelion workflows. Analyses included:
- Clonotype assignment
- Clonal expansion analysis
- Clone size categorisation
- Repertoire diversity assessment
- Transcriptome-repertoire integration

### 7. Statistical visualisation
Results were visualised using Python and R-based plotting workflows including UMAP visualisation, differential abundance plots and clonotype distribution analyses.

## Infrastructure

Analyses were executed in Linux-based HPC environments using:
- bash
- conda environments
- Jupyter notebooks
- Python and R workflows

## Notes

Raw sequencing datasets and sensitive patient data are not included in this repository.

This repository focuses on reproducible downstream computational workflows and analyses.
