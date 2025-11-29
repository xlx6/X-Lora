#!/usr/bin/env bash
set -euo pipefail

MODEL=./distilbert-base-uncased
DATA_DIR=./sst2
EPOCHS=3
BATCH=256
LR=2e-5

mkdir -p outputs/exp1 outputs/exp2 outputs/exp3

echo "[Exp1] Fixed rank r=8 across methods"
R=8
python train.py --method baseline --rank $R --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp1/baseline_r${R}
python train.py --method svd      --rank $R --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp1/svd_r${R}
python train.py --method ortho    --rank $R --orthogonal_lambda 1e-4 --use_qr --qr_frequency 10 --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp1/ortho_r${R}_lam1e-4
python train.py --method struct   --rank $R --structure_type lu --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp1/struct_r${R}

echo "[Exp2] Parameter efficiency sweep over r"
for R in 2 4 8 16 32 64; do
  python train.py --method baseline --rank $R --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp2/baseline_r${R}
  python train.py --method svd      --rank $R --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp2/svd_r${R}
  python train.py --method ortho    --rank $R --orthogonal_lambda 1e-4 --use_qr --qr_frequency 10 --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp2/ortho_r${R}_lam1e-4
  python train.py --method struct   --rank $R --structure_type lu --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp2/struct_r${R}
done

echo "[Exp3] Ortho ablation over lambda at r=8"
R=8
for LAM in 0 1e-5 1e-4 1e-3 1e-2; do
  python train.py --method ortho --rank $R --orthogonal_lambda $LAM --use_qr --qr_frequency 10 --model_name_or_path $MODEL --epochs $EPOCHS --batch_size $BATCH --learning_rate $LR --output_dir outputs/exp3/ortho_r${R}_lam${LAM}
done

echo "Aggregating results to outputs/summary.csv"
python tools/aggregate_results.py --root outputs --out outputs/summary.csv

echo "Done."

