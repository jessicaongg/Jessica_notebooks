# Single-cell & TCR Analysis Workflow

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

## Workflow Components

### Single-cell preprocessing
- Cell and gene quality control
- Filtering low-quality cells
- Normalisation and scaling

### Batch integration
- scVI latent embedding
- Batch correction across samples

### Clustering and annotation
- Leiden clustering
- UMAP dimensionality reduction
- Canonical marker-based annotation

### Differential abundance analysis
- Milo neighbourhood testing
- scCODA compositional modelling
- Age and sex covariate modelling

### TCR repertoire analysis
- Dandelion integration
- Clonotype assignment
- Clone size categorisation
- Repertoire diversity analysis

### Downstream visualisation
- UMAP visualisation
- Differential abundance plots
- Clonotype distribution visualisation
