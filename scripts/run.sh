#!/bin/bash
set -e

echo "Running CD4 Milo..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD4/03_PICA_CD4_milo_pertpy.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD4 milo done"

echo "Running CD4 Milo age..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD4/03_PICA_CD4_milo_pertpy_age.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD4 milo age done"

echo "Running CD4 Milo sex..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD4/03_PICA_CD4_milo_pertpy_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD4 milo sex done"

echo "Running CD4 Milo interaction..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/CD4/03_PICA_CD4_milo_pertpy_age_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "CD4 milo interaction done"

echo "CD4 All done!"