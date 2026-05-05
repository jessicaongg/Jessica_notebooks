#!/usr/bin/env bash
set -euo pipefail

OUTDIR="/scratch/user/$USER/BIOX7014_Thesis/data/PICA_Batch001-Batch007_tcr_merged"
mkdir -p "$OUTDIR"

batches=(
"/QRISdata/Q7556/20240530_WGS_20240530_sc_PICA0001-PICA0007_PMID_97-101"
"/QRISdata/Q7556/20240918_WGS_20240924_sc_PICA0008-PICA0032"
"/QRISdata/Q7556/20241106_WGS_20241106_sc_PICA0033-PICA0069"
"/QRISdata/Q7556/20250114_WGS_20241218_sc_PICA0071-PICA0097"
"/QRISdata/Q7556/20250324_WGS_20250324_sc_PICA0098-PICA0118"
"/QRISdata/Q7556/20250612_WGS_20250612_sc_PICA0119-PICA0146"
"/QRISdata/Q7556/20250814_WGS_20250814_sc_PICA0147-PICA0173"
)

rewrite_stream_to_out() {
    local prefix="$1"
    local keep_header="$2"
    local outfile="$3"

    awk -v prefix="$prefix" -v keep_header="$keep_header" '
    BEGIN { FS=OFS="," }
    NR==1 {
        barcode_col=0
        for (i=1; i<=NF; i++) {
            if ($i=="barcode") {
                barcode_col=i
                break
            }
        }
        if (barcode_col==0) {
            print "ERROR: barcode column not found" > "/dev/stderr"
            exit 1
        }
        if (keep_header==1) print
        next
    }
    {
        $barcode_col = prefix $barcode_col
        print
    }' >> "$outfile"
}

for d in "${batches[@]}"; do
    batch=$(basename "$d")
    out="$OUTDIR/${batch}_filtered_contig_annotations.csv"

    echo "=== Processing $batch ==="
    rm -f "$out"
    first=1
    found=0

    # tar-based batches
    if compgen -G "$d/*_Pool_*.tar" > /dev/null; then
        for f in "$d"/*_Pool_*.tar; do
            pool=$(basename "$f" .tar)
            prefix="${pool}_"
            echo "Reading tar: $(basename "$f")"

            if [[ $first -eq 1 ]]; then
                tar -xf "$f" -O --wildcards '*/vdj_t/filtered_contig_annotations.csv' \
                  | rewrite_stream_to_out "$prefix" 1 "$out"
                first=0
            else
                tar -xf "$f" -O --wildcards '*/vdj_t/filtered_contig_annotations.csv' \
                  | rewrite_stream_to_out "$prefix" 0 "$out"
            fi
            found=1
        done

    # non-tar batches: prefer 02_cellranger_results, fallback to vdj_t
    else
        if find -L "$d" -path '*/02_cellranger_results/filtered_contig_annotations.csv' | grep -q .; then
            pattern='*/02_cellranger_results/filtered_contig_annotations.csv'
        else
            pattern='*/vdj_t/filtered_contig_annotations.csv'
        fi

        while IFS= read -r f; do
            echo "Reading file: $f"

            pool=$(echo "$f" | grep -o 'Pool[_-][0-9]\+' | head -n1 | sed 's/-/_/g')
            [[ -z "$pool" ]] && pool="Pool_unknown"

            prefix="${batch}_${pool}_"

            if [[ $first -eq 1 ]]; then
                cat "$f" | rewrite_stream_to_out "$prefix" 1 "$out"
                first=0
            else
                cat "$f" | rewrite_stream_to_out "$prefix" 0 "$out"
            fi
            found=1
        done < <(find -L "$d" -path "$pattern" | sort)
    fi

    if [[ $found -eq 0 ]]; then
        echo "WARNING: No filtered_contig_annotations.csv found for $batch"
    else
        echo "Wrote $out"
        wc -l "$out"
    fi
done

echo "Done"

for f in /scratch/user/$USER/BIOX7014_Thesis/data/PICA_Batch001-Batch007_tcr_merged/*_filtered_contig_annotations.csv; do

    echo "=== Checking $(basename "$f") ==="
    awk -F, '
    NR==1 {
        for (i=1; i<=NF; i++) if ($i=="barcode") bc=i
        next
    }
    { print $bc }
    ' "$f" | sort | uniq -d | wc -l
done