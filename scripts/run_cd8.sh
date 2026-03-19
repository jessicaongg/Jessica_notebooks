#!/bin/bash
set -e

echo "Running CD8 Milo..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD8/03_PICA_CD8_milo_pertpy.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD8 milo done"

echo "Running CD8 Milo age..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD8/03_PICA_CD8_milo_pertpy_age.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD8 milo age done"

echo "Running CD8 Milo sex..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD8/03_PICA_CD8_milo_pertpy_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD8 milo sex done"

echo "Running CD8 Milo interaction..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD8/03_PICA_CD8_milo_pertpy_age_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD8 milo interaction done"

echo "CD8 All done!"