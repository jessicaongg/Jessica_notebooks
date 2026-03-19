#!/bin/bash
set -e

echo "Running NK Milo..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/NK/03_PICA_NK_milo_pertpy.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "NK milo done"

echo "Running NK Milo age..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/NK/03_PICA_NK_milo_pertpy_age.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "NK milo age done"

echo "Running NK Milo sex..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/NK/03_PICA_NK_milo_pertpy_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "NK milo sex done"

echo "Running NK Milo interaction..."
python -m nbconvert --to notebook --execute --inplace "/scratch/user/s4575250/BIOX7014_Thesis/notebooks/03_abundance_change/PICA Batch001-Batch007/NK/03_PICA_NK_milo_pertpy_age_sex.ipynb" --ExecutePreprocessor.timeout=-1 --log-level=INFO
echo "NK milo interaction done"

echo "NK All done!"