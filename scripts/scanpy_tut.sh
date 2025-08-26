###############################################################################
########### Preprocessing and clustering scRNA-seq data with Scanpy ###########
###############################################################################

## Core libraries
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#data retrieval
import pooch

# set up how plots are displayed
#lower dpi = lower resolution (faster rendering)
# higher dpi for exporting figures
# facecolor is the background color of the figure
sc.set_figure_params(dpi=50, facecolor="white") 

# create pooch object (download manager) to retrieve example data
# this is a local cache, so it will not download the data again if it already exists. # sets where the data will be stored
# the path is set to the cache directory for scverse tutorials
EG_DATA = pooch.create(
    path=pooch.os_cache("scverse_tutorials"),
    base_url="doi:10.6084/m9.figshare.22716739.v1/",   # the base_url is a DOI that points to the data on figshare
)
EG_DATA.load_registry_from_doi().  # loads the registry from the DOI, which contains the metadata for the files. 
# ensures that files downloaded from corect location # data integrity: match exoected filenames and checksums (hashes) 


## load scRNA-seq data (HDF5 files) for 2 samples
# the data is in 10x Genomics format (HDF5), which is a common format for storing large scRNA-seq data
# the data is stored in a dictionary,keys:sample IDs, values: HDF5 filenames

# the expression matrix is a sparse matrix, where the rows are the genes and the columns are the cells
# the cell metadata is stored in the `obs` attribute of the AnnData object
# the gene metadata is stored in the `var` attribute of the AnnData object  
samples = {
    "s1d1": "s1d1_filtered_feature_bc_matrix.h5",
    "s1d3": "s1d3_filtered_feature_bc_matrix.h5",
}

# create an empty dictionary to store the AnnData objects for each sample
adatas = {}

# the data is loaded into an AnnData object, which is a data structure used by Scanpy to store single-cell data
# the AnnData object contains the expression matrix, the cell metadata, and the gene metadata
# loop through the samples and load the data into the AnnData objects
for sample_id, filename in samples.items():
    path = EG_DATA.fetch(filename)              # Download the file from the cache or the remote location(figshare)
    sample_adata = sc.read_10x_h5(path)         # Read the 10x Genomics HDF5 file into an AnnData object
    sample_adata.var_names_make_unique()        # Ensure that gene names are unique across the dataset (fix duplicates)
    adatas[sample_id] = sample_adata            # Store the AnnData object in the dictionary with the sample ID as the key

adata = ad.concat(adatas, label="sample")      # Concatenate the AnnData objects into a single AnnData object # add new column "sample" to obs
adata.obs_names_make_unique()                  # Ensure that cell names/barcodes are unique across the dataset
print(adata.obs["sample"].value_counts())      # no. of cells per sample
adata  # Return object

###############################################################################
######################### Quality control and filtering #######################
###############################################################################

#.var is a dataframe that contains the gene metadata
#.obs is a dataframe that contains the cell metadata
adata.var["mt"] = adata.var_names.str.startswith("MT-") # MT for humans and Mt for mice
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))  #small and large subunits
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]") # gene starting with HB but not HB-P (pseudogene)

# Calculate per cell QC metrics and add them to .obs dataframe
# this is done to filter out low-quality cells based on the number of genes expressed, total counts, and percentage of mitochondrial/ribosomal/hemoglobin gene
# sc = scanpy, pp = preprocessing module (submodule of scanpy) that include filtering, normalization, and scaling functions
sc.pp.calculate_qc_metrics(                                                            
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)
# inplace=True: modifies the adata object directly
# log1p=True: applies log transformation to the QC metrics (log(x+1))

# After running this, you’ll see columns like:

# Column name	          Description
# total_counts	        Total UMI counts per cell
# #n_genes_by_counts	Number of genes expressed (non-zero counts)
# log1p_total_counts	log(1 + total_counts)
# pct_counts_mt	        % of total counts from mitochondrial genes
# pct_counts_ribo	    % from ribosomal genes
# pct_counts_hb	        % from hemoglobin genes

######################### Visualize the QC metrics using violin plots##########

# pl = plotting module (submodule of scanpy) that includes functions for visualizing data
# this will help to identify outliers and cells with low quality
# the violin plot shows the distribution of the QC metrics for each cell                       
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"],
    jitter=0.4,             # jitter=0.4: adds random noise to the data points to make them more visible
    multi_panel=True,        # multi_panel=True: creates a separate panel for each variable
)

# Scatter plot to visualize the relationship between total counts and number of genes expressed
for gene_type in ["mt", "ribo", "hb"]:
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        color=f"pct_counts_{gene_type}",
        title=f"Total counts vs. n_genes_by_counts for {gene_type.upper()} genes"
    )
######################## Filter cells based on QC metrics######################

# Remove emtpy droplets/low-quality cells and uninformative genes
sc.pp.filter_cells(adata, min_genes=100)  # filter cells with less than 100 genes expressed(low quality cells)
sc.pp.filter_genes(adata, min_cells=3)    # filter genes expressed in less than 3 cells (low quality genes)

# OR

# Filter based on mt/ribo/hb gene expression
# Set QC thresholds (adjust if needed)
min_genes = 200
max_genes = 6000
max_counts = 50000
max_mt = 15          # mitochondrial %
max_ribo = 60        # ribosomal %
max_hb = 10          # hemoglobin %

# Boolean mask for filtering
adata = adata[ 
    (adata.obs['n_genes_by_counts'] >= min_genes) &
    (adata.obs['n_genes_by_counts'] <= max_genes) &
    (adata.obs['total_counts'] <= max_counts) &
    (adata.obs['pct_counts_mt'] < max_mt) &
    (adata.obs['pct_counts_ribo'] < max_ribo) &
    (adata.obs['pct_counts_hb'] < max_hb), :
]
########################## Doublet detection and removal ######################
# Doublets are cells that contain two or more cells in a single droplet
# They can be detected using the Scrublet algorithm, which uses the expression profile of each
sc.pp.scrublet(adata, batch_key="sample")

###############################################################################
######################### Normalisation #######################################
###############################################################################
